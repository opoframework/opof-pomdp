#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <cstring>
#include <limits>
#include <magic/core/Belief.h>
#include <magic/core/Types.h>
#include <magic/core/planning/DespotPlanner.h>
#include <magic/core/simulations/LightDark.h>
#include <magic/core/simulations/PuckPush.h>
#include <magic/core/simulations/VdpTag.h>
#include <numpy/arrayobject.h>

using namespace simulations;

/* ================================
 * ===== Conversion Utilities =====
 * ================================
 */
std::string as_string(PyObject *x) {
  return std::string(PyBytes_AsString(PyUnicode_AsUTF8String(PyObject_Str(x))));
}

PyObject *to_string(const std::string &s) {
  return PyUnicode_DecodeUTF8(s.c_str(), s.length(), nullptr);
}

// According to the source code,
// https://github.com/python/cpython/blob/1b55b65638254aa78b005fbf0b71fb02499f1852/Objects/dictobject.c#L1532,
// values are incref'ed which means we need to decref once on the inputs.
void decref_dict(PyObject *dict) {
  PyObject *keys = PyDict_Keys(dict);
  for (size_t i = 0; i < PyList_Size(keys); i++) {
    Py_DECREF(PyDict_GetItem(dict, PyList_GetItem(keys, i)));
  }
  Py_DECREF(keys);
}

PyObject *to_list(const std::vector<float> &d) {
  PyObject *list = PyList_New(d.size());
  for (size_t i = 0; i < d.size(); i++) {
    PyList_SetItem(list, i, PyFloat_FromDouble(d[i]));
  }
  return list;
}

std::vector<float> to_vec(PyObject *list) {
  std::vector<float> v(PyList_Size(list));
  for (size_t i = 0; i < v.size(); i++) {
    v[i] = static_cast<float>(PyFloat_AsDouble(PyList_GetItem(list, i)));
  }
  return v;
}

template <typename ExpSimulation>
PyObject *to_numpy(const Belief<ExpSimulation> &b) {
  PyObject *list = nullptr;
  float *buffer = nullptr;
  for (size_t i = 0; i < b._particles.size(); i++) {
    // Write particle data.
    std::vector<float> particle_data;
    b._particles[i].Encode(particle_data);
    // Create np array lazily from dimensions.
    if (list == nullptr) {
      npy_intp dims[] = {static_cast<npy_intp>(b._particles.size()),
                         static_cast<npy_intp>(particle_data.size())};
      list = PyArray_SimpleNew(2, dims, NPY_FLOAT);
      buffer = (float *)PyArray_DATA((PyArrayObject *)list);
    }
    // Copy particle to np array.
    for (size_t j = 0; j < particle_data.size(); j++) {
      *((float *)PyArray_GETPTR2((PyArrayObject *)list, i, j)) =
          particle_data[j];
    }
  }
  return list;
}

template <typename ExpSimulation>
Belief<ExpSimulation> to_belief(PyObject *list) {
  Belief<ExpSimulation> belief;
  for (size_t i = 0; i < belief._particles.size(); i++) {
    std::vector<float> particle_data(PyArray_DIM((PyArrayObject *)list, 1));
    for (size_t j = 0; j < particle_data.size(); j++) {
      particle_data[j] =
          *((float *)PyArray_GETPTR2((PyArrayObject *)list, i, j));
    }
    belief._particles[i].Decode(particle_data);
  }
  return belief;
}

template <typename ExpSimulation>
PyObject *to_list(const std::vector<typename ExpSimulation::Action> &actions) {
  PyObject *list = PyList_New(actions.size());
  for (size_t i = 0; i < actions.size(); i++) {
    std::vector<float> action_data;
    actions[i].Encode(action_data);
    PyList_SetItem(list, i, to_list(action_data));
  }
  return list;
}

template <typename ExpSimulation>
std::vector<typename ExpSimulation::Action> to_macro_action(PyObject *list) {
  std::vector<typename ExpSimulation::Action> macro_action(PyList_Size(list));
  for (size_t i = 0; i < macro_action.size(); i++) {
    std::vector<float> action_data = to_vec(PyList_GetItem(list, i));
    macro_action[i].Decode(action_data);
  }
  return macro_action;
}

PyObject *to_numpy(cv::Mat img) {
  // Prepare dimensions and buffer.
  size_t s = 1;
  npy_intp *dim = new npy_intp[img.size.dims() + 1];
  for (size_t i = 0; i < img.size.dims(); i++) {
    s *= img.size.p[i];
    dim[i] = img.size.p[i];
  }
  dim[img.size.dims()] = img.channels();
  s *= img.channels();

  // Create numpy array.
  PyObject *mat = PyArray_SimpleNew(img.size.dims() + 1, dim, NPY_UINT8);

  // Copy data.
  std::memcpy((uchar *)PyArray_DATA((PyArrayObject *)mat), img.data,
              s * sizeof(uchar));

  delete[] dim;
  return mat;
}

/* ===========================
 * ===== rand() Function =====
 * ===========================
 */

template <typename ExpSimulation> PyObject *rand_impl() {
  // Randomize context, state, belief.
  auto sim = ExpSimulation::CreateRandom();
  auto belief = Belief<ExpSimulation>::FromInitialState();

  // Get context data.
  std::vector<float> context_data;
  sim.EncodeContext(context_data);

  // Get state data.
  std::vector<float> state_data;
  sim.Encode(state_data);

  // Write and return result.
  PyObject *result = PyTuple_New(3);
  PyTuple_SetItem(result, 0, to_list(context_data));
  PyTuple_SetItem(result, 1, to_list(state_data));
  PyTuple_SetItem(result, 2, to_numpy(belief));
  return result;
}

PyObject *rand(PyObject *self, PyObject *args) {
  PyObject *task;
  PyArg_UnpackTuple(args, "ref", 1, 1, &task);

  if (as_string(task) == "LightDark") {
    return rand_impl<LightDark>();
  } else if (as_string(task) == "PuckPush") {
    return rand_impl<PuckPush>();
  } else if (as_string(task) == "VdpTag") {
    return rand_impl<VdpTag>();
  } else {
    throw std::invalid_argument("Unsupported task");
  }
}

/* ============================
 * ===== solve() Function =====
 * ============================
 */

template <typename ExpSimulation>
PyObject *solve_impl(PyObject *context, PyObject *belief,
                     PyObject *macro_action_params,
                     PyObject *macro_action_length) {
  // Set context globally.
  ExpSimulation::DecodeContext(to_vec(context));

  // Extract macro-actions.
  std::vector<float> _macro_action_params = to_vec(macro_action_params);
  size_t _macro_action_length = PyLong_AsLong(macro_action_length);
  auto macro_actions = ExpSimulation::Action::Deserialize(_macro_action_params,
                                                          _macro_action_length);

  // Plan.
  planning::DespotPlanner<ExpSimulation, Belief<ExpSimulation>> planner;
  auto plan_result = planner.Search(
      to_belief<ExpSimulation>(belief).ResampleNonTerminal(), macro_actions);
  PyObject *result = PyDict_New();
  PyDict_SetItem(result, to_string("value"),
                 PyFloat_FromDouble(plan_result.value));
  PyDict_SetItem(result, to_string("depth"),
                 PyLong_FromLong(plan_result.depth));
  PyDict_SetItem(result, to_string("num_nodes"),
                 PyLong_FromLong(plan_result.num_nodes));
  PyDict_SetItem(result, to_string("action"),
                 to_list<ExpSimulation>(plan_result.action));
  decref_dict(result);
  return result;
}

static PyObject *solve(PyObject *self, PyObject *args) {
  PyObject *task;
  PyObject *context;
  PyObject *belief;
  PyObject *macro_action_params;
  PyObject *macro_action_length;
  PyArg_UnpackTuple(args, "ref", 5, 5, &task, &context, &belief,
                    &macro_action_params, &macro_action_length);

  if (as_string(task) == "LightDark") {
    return solve_impl<LightDark>(context, belief, macro_action_params,
                                 macro_action_length);
  } else if (as_string(task) == "PuckPush") {
    return solve_impl<PuckPush>(context, belief, macro_action_params,
                                macro_action_length);
  } else if (as_string(task) == "VdpTag") {
    return solve_impl<VdpTag>(context, belief, macro_action_params,
                              macro_action_length);
  } else {
    throw std::invalid_argument("Unsupported task");
  }
}

/* ===========================
 * ===== step() Function =====
 * ===========================
 */

template <typename ExpSimulation>
PyObject *step_impl(PyObject *context, PyObject *state, PyObject *belief,
                    PyObject *macro_action, PyObject *render,
                    PyObject *macro_action_params) {
  // Set context.
  ExpSimulation::DecodeContext(to_vec(context));

  // Create state.
  auto sim = ExpSimulation();
  sim.Decode(to_vec(state));

  // Get belief.
  Belief<ExpSimulation> _belief = to_belief<ExpSimulation>(belief);
  _belief = _belief.ResampleNonTerminal();

  // Get action.
  auto _macro_action = to_macro_action<ExpSimulation>(macro_action);

  // Propagate sim.
  auto execution_result = planning::
      DespotPlanner<ExpSimulation, Belief<ExpSimulation>>::ExecuteMacroAction(
          sim, _macro_action, ExpSimulation::MAX_STEPS - sim.step);

  // Prepare origin and macro actions for rendering.
  vector_t macro_action_start;
  std::vector<std::vector<typename ExpSimulation::Action>> macro_actions;
  if (PyObject_IsTrue(render)) {
    macro_action_start = sim.ego_agent_position;
    std::vector<float> _macro_action_params = to_vec(macro_action_params);
    macro_actions = ExpSimulation::Action::Deserialize(_macro_action_params,
                                                       _macro_action.size());
  }

  // Propagate belief.
  float min_belief_error = std::numeric_limits<float>::infinity();
  PyObject *frames = PyList_New(0);
  for (size_t i = 0;
       i < execution_result.state_trajectory.size() && !_belief.IsTerminal();
       i++) {
    // Optionally render.
    if (PyObject_IsTrue(render)) {
      std::vector<ExpSimulation> belief_sample;
      for (size_t j = 0; j < 1000; j++) {
        belief_sample.emplace_back(_belief.Sample());
      }
      auto frame = sim.Render(belief_sample, macro_actions, macro_action_start);
      PyList_Append(frames, to_numpy(frame));
    }

    _belief.Update(_macro_action[i],
                   execution_result.observation_trajectory[i]);
    sim = execution_result.state_trajectory[i];
    min_belief_error = std::min(min_belief_error, _belief.Error(sim));
  }

  PyObject *result = PyDict_New();
  PyDict_SetItem(result, to_string("terminal"),
                 PyFloat_FromDouble(_belief.IsTerminal() || sim.IsTerminal()));
  // Failure if actual state is failure, or if actual state is non-failure
  // but belief is in failure.
  PyDict_SetItem(result, to_string("failure"),
                 PyFloat_FromDouble(sim.IsFailure() || (_belief.IsTerminal() &&
                                                        !sim.IsTerminal())));
  PyDict_SetItem(result, to_string("reward"),
                 PyFloat_FromDouble(execution_result.undiscounted_reward));
  PyDict_SetItem(result, to_string("steps"),
                 PyFloat_FromDouble(execution_result.state_trajectory.size()));
  PyDict_SetItem(result, to_string("min_belief_error"),
                 PyFloat_FromDouble(min_belief_error));
  std::vector<float> state_data;
  sim.Encode(state_data);
  PyDict_SetItem(result, to_string("state"), to_list(state_data));
  PyDict_SetItem(result, to_string("belief"), to_numpy(_belief));
  PyDict_SetItem(result, to_string("frames"), frames);
  decref_dict(result);
  return result;
}

static PyObject *step(PyObject *self, PyObject *args) {
  PyObject *task;
  PyObject *context;
  PyObject *state;
  PyObject *belief;
  PyObject *macro_action;
  PyObject *render;
  PyObject *render_macro_actions;

  PyArg_UnpackTuple(args, "ref", 7, 7, &task, &context, &state, &belief,
                    &macro_action, &render, &render_macro_actions);
  if (as_string(task) == "LightDark") {
    return step_impl<LightDark>(context, state, belief, macro_action, render,
                                render_macro_actions);
  } else if (as_string(task) == "PuckPush") {
    return step_impl<PuckPush>(context, state, belief, macro_action, render,
                               render_macro_actions);
  } else if (as_string(task) == "VdpTag") {
    return step_impl<VdpTag>(context, state, belief, macro_action, render,
                             render_macro_actions);
  } else {
    throw std::invalid_argument("Unsupported task");
  }
}

// Exported methods are collected in a table
PyMethodDef methods[] = {
    {"rand", (PyCFunction)rand, METH_VARARGS,
     "Creates a POMDP task with a random context, initial state, and belief"},
    {"solve", (PyCFunction)solve, METH_VARARGS,
     "Solves a POMDP problem given a current belief"},
    {"step", (PyCFunction)step, METH_VARARGS,
     "Applies a macro-action to a POMDP problem, propagating its state and an "
     "associated belief"},
    {NULL, NULL, 0, NULL} // Sentinel value ending the table
};

// A struct contains the definition of a module
PyModuleDef module = {
    PyModuleDef_HEAD_INIT,
    "pomdp_core", // Module name
    "Module containing compiled codes to run POMDP planners",
    -1, // Optional size of the module state memory
    methods,
    NULL, // Optional slot definitions
    NULL, // Optional traversal function
    NULL, // Optional clear function
    NULL  // Optional module deallocation function
};

// The module init function
PyMODINIT_FUNC PyInit_pomdp_core(void) {
  import_array(); // Without these the numpy stuffs will segfault.
  return PyModule_Create(&module);
}
