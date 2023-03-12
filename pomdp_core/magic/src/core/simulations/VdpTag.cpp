#include "magic/core/simulations/VdpTag.h"

#include "magic/core/Util.h"
#include <random>

namespace simulations {

VdpTag::Action VdpTag::Action::Rand() {
  Action action(std::uniform_real_distribution<float>(0, 2 * PI)(Rng()));
  action.look = std::bernoulli_distribution(0.5)(Rng());
  return action;
}

uint64_t VdpTag::Observation::Discretize() const {
  list_t<int> data;
  for (size_t i = 0; i < beam_distances.size(); i++) {
    data.emplace_back(static_cast<int>(floorf(beam_distances[i] / 0.1)));
  }
  return boost::hash_value(data);
}

list_t<list_t<VdpTag::Action>>
VdpTag::Action::CreateHandcrafted(size_t length) {
  list_t<list_t<Action>> macro_actions;
  for (size_t i = 0; i < 8; i++) {
    macro_actions.emplace_back();
    for (size_t j = 0; j < length; j++) {
      macro_actions.back().push_back({static_cast<float>(i) * 2 * PI / 8});
    }
  }
  Action trigger_action;
  trigger_action.look = true;
  macro_actions.emplace_back();
  macro_actions.back().emplace_back(trigger_action);
  return macro_actions;
}

list_t<list_t<VdpTag::Action>>
VdpTag::Action::Deserialize(const list_t<float> &params, size_t macro_length) {
  list_t<list_t<VdpTag::Action>> macro_actions =
      StandardMacroActionDeserialization<VdpTag::Action>(params, macro_length);
  Action look_action;
  look_action.look = true;
  macro_actions.emplace_back();
  macro_actions.back().emplace_back(look_action);
  return macro_actions;
}

void VdpTag::Action::Encode(list_t<float> &data) const {
  data.emplace_back(look ? 1.0 : 0.0);
  data.emplace_back(angle);
}

void VdpTag::Action::Decode(const list_t<float> &data) {
  look = static_cast<size_t>(std::lround(data[0])) == 1;
  angle = data[1];
}

/* ====== Construction related functions ====== */

VdpTag::VdpTag() : step(0), _is_terminal(false), _is_failure(false) {}

VdpTag VdpTag::CreateRandom() { return SampleBeliefPrior(); }

/* ====== Belief related functions ====== */
VdpTag VdpTag::SampleBeliefPrior() {
  VdpTag sim;
  sim.ego_agent_position = vector_t(0.0f, 0.0f);
  sim.exo_agent_position.x =
      std::uniform_real_distribution<float>(-4.0, 4.0)(Rng());
  sim.exo_agent_position.y =
      std::uniform_real_distribution<float>(-4.0, 4.0)(Rng());
  return sim;
}

float VdpTag::Error(const VdpTag &other) const {
  float error = 0;
  error += (ego_agent_position - other.ego_agent_position).norm();
  error += (exo_agent_position - other.exo_agent_position).norm();
  return error / 2;
}

/* ====== Bounds related functions ====== */
float VdpTag::BestReward() const {
  if (_is_terminal) {
    return 0;
  } // Return value of 0 needed for DESPOT.

  float distance = std::max(
      0.0f, (exo_agent_position - ego_agent_position).norm() - TAG_RADIUS);
  float max_distance_per_step =
      AGENT_SPEED * DELTA + 4.0f * DELTA + 3 * POS_STD;
  size_t steps =
      static_cast<size_t>(round(ceilf(distance / max_distance_per_step)));
  if (steps <= 1) {
    return TAG_REWARD;
  } else {
    return (1 - powf(GAMMA, static_cast<float>(steps) - 1)) /
               (1 - static_cast<float>(steps)) * STEP_REWARD +
           powf(GAMMA, static_cast<float>(steps) - 1) * TAG_REWARD;
  }
}

/* ====== Stepping functions ====== */

float VdpTag::Cross(const vector_t &a, const vector_t &b) {
  return a.x * b.y - b.x * a.y;
}

vector_t VdpTag::VdpDynamics(const vector_t &v) const {
  return {MU * (v.x - v.x * v.x * v.x / 3 - v.y), v.x / MU};
}

vector_t VdpTag::Rk4Step(const vector_t &v) const {
  float h = RK4_STEP_SIZE;
  vector_t k1 = VdpDynamics(v);
  vector_t k2 = VdpDynamics(v + k1 * h / 2);
  vector_t k3 = VdpDynamics(v + k2 * h / 2);
  vector_t k4 = VdpDynamics(v + k3 * h);
  return v + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4);
}

vector_t VdpTag::BarrierStop(const vector_t &v, const vector_t &d) const {
  float shortest_u = 1.0f + 2 * std::numeric_limits<float>::epsilon();
  vector_t q = v;
  vector_t s = d;

  for (vector_t dir : list_t<vector_t>{{1, 0}, {0, 1}, {-1, 0}, {0, -1}}) {
    vector_t p = 0.2f * dir;
    vector_t r = 2.8f * dir;
    float rxs = Cross(r, s);
    if (rxs == 0.0f) {
      continue;
    } else {
      vector_t qmp = q - p;
      float u = Cross(qmp, r) / rxs;
      float t = Cross(qmp, s) / rxs;
      if (0.0f <= u && u < shortest_u && 0.0f <= t && t <= 1.0f) {
        shortest_u = u;
      }
    }
  }

  return v + (shortest_u - 2 * std::numeric_limits<float>::epsilon()) * d;
}

size_t VdpTag::ActiveBeam(const vector_t &v) const {
  float angle = AngleTo(vector_t(1.0f, 0.0f), v);
  while (angle <= 0.0f) {
    angle += 2 * PI;
  }
  size_t x = static_cast<size_t>(lround(ceilf(8 * angle / (2 * PI))) - 1);
  return std::max(static_cast<size_t>(0), std::min(static_cast<size_t>(7), x));
}

template <bool compute_log_prob>
std::tuple<VdpTag, float, VdpTag::Observation, float>
VdpTag::Step(const VdpTag::Action &action,
             const VdpTag::Observation *observation) const {
  if (_is_terminal) {
    throw std::logic_error("Cannot step terminal simulation.");
  }

  VdpTag next_sim = *this;
  float reward = 0;

  /* ====== Step 1: Update state. ====== */
  if (!action.look) {
    next_sim.ego_agent_position =
        BarrierStop(next_sim.ego_agent_position,
                    AGENT_SPEED * DELTA * vector_t(1, 0).rotated(action.angle));
  }

  for (size_t i = 0; i < RK4_STEP_ITER; i++) {
    next_sim.exo_agent_position = Rk4Step(next_sim.exo_agent_position);
  }
  next_sim.exo_agent_position.x +=
      std::normal_distribution<float>(0.0, POS_STD)(RngDet());
  next_sim.exo_agent_position.y +=
      std::normal_distribution<float>(0.0, POS_STD)(RngDet());
  next_sim.step++;

  // Check terminal and rewards.
  if ((next_sim.ego_agent_position - next_sim.exo_agent_position).norm() <
      TAG_RADIUS) {
    reward = TAG_REWARD;
    next_sim._is_terminal = true;
  } else {
    reward = STEP_REWARD;
  }
  if (action.look) {
    reward += ACTIVE_MEAS_REWARD;
  }

  if (!_is_terminal) {
    if (next_sim.step == MAX_STEPS) {
      next_sim._is_terminal = true;
      next_sim._is_failure = true;
    }
  }

  /* ====== Step 2: Generate observation. ====== */
  Observation new_observation;
  if (observation) {
    new_observation = *observation;
  }
  float log_prob = 0;
  vector_t rel_pos = next_sim.exo_agent_position - next_sim.ego_agent_position;
  float dist = rel_pos.norm();
  size_t active_beam = ActiveBeam(rel_pos);
  if (action.look) {
    if (!observation) {
      new_observation.beam_distances[active_beam] =
          std::normal_distribution<float>(dist, ACTIVE_MEAS_STD)(RngDet());
    }
    if constexpr (compute_log_prob) {
      log_prob += NormalLogProb(dist, ACTIVE_MEAS_STD,
                                new_observation.beam_distances[active_beam]);
    }
  } else {
    if (!observation) {
      new_observation.beam_distances[active_beam] =
          std::normal_distribution<float>(dist, MEAS_STD)(RngDet());
    }
    if constexpr (compute_log_prob) {
      log_prob += NormalLogProb(dist, MEAS_STD,
                                new_observation.beam_distances[active_beam]);
    }
  }
  for (size_t i = 0; i < new_observation.beam_distances.size(); i++) {
    if (i != active_beam) {
      if (!observation) {
        new_observation.beam_distances[i] =
            std::normal_distribution<float>(1.0, MEAS_STD)(RngDet());
      }
      if constexpr (compute_log_prob) {
        log_prob +=
            NormalLogProb(1.0, MEAS_STD, new_observation.beam_distances[i]);
      }
    }
  }

  return std::make_tuple(next_sim, reward,
                         observation ? Observation() : new_observation,
                         log_prob);
}
template std::tuple<VdpTag, float, VdpTag::Observation, float>
VdpTag::Step<true>(const VdpTag::Action &action,
                   const VdpTag::Observation *observation) const;
template std::tuple<VdpTag, float, VdpTag::Observation, float>
VdpTag::Step<false>(const VdpTag::Action &action,
                    const VdpTag::Observation *observation) const;

/* ====== Serialization functions ====== */

void VdpTag::Encode(list_t<float> &data) const {
  data.emplace_back(static_cast<float>(step));
  ego_agent_position.Encode(data);
  exo_agent_position.Encode(data);
}

void VdpTag::Decode(const list_t<float> &data) {
  auto iter = data.begin();
  step = static_cast<size_t>(std::lround(*(iter++)));
  ego_agent_position = Vector2D::Decode(iter);
  exo_agent_position = Vector2D::Decode(iter);
}

void VdpTag::EncodeContext(list_t<float> &data) {}

void VdpTag::DecodeContext(const list_t<float> &data) {}

cv::Mat VdpTag::Render(const list_t<VdpTag> &belief_sims,
                       const list_t<list_t<Action>> &macro_actions,
                       const vector_t &macro_action_start) const {}

} // namespace simulations
