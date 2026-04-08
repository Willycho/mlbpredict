#include "sim_core.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>

namespace py = pybind11;
namespace sim {

// ============================================================
// FastRng (xoshiro256+)
// ============================================================

uint64_t FastRng::splitmix64(uint64_t& x) {
    uint64_t z = (x += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

FastRng::FastRng(uint64_t seed) {
    s[0] = splitmix64(seed);
    s[1] = splitmix64(seed);
    s[2] = splitmix64(seed);
    s[3] = splitmix64(seed);
}

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

double FastRng::next_double() {
    const uint64_t result = s[0] + s[3];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return (result >> 11) * 0x1.0p-53;  // [0, 1)
}

uint32_t FastRng::next_uint32() {
    return static_cast<uint32_t>(next_double() * 4294967296.0);
}

// ============================================================
// resolve_pa
// ============================================================

Event resolve_pa(const MatchupProbs& probs, FastRng& rng) {
    double r = rng.next_double();

    // Stage 1: K, BB, HBP, HR, BIP
    float cum = 0.0f;
    for (int i = 0; i < NUM_STAGE1 - 1; i++) {
        cum += probs.stage1[i];
        if (r < cum) {
            return static_cast<Event>(i);  // K=0, BB=1, HBP=2, HR=3
        }
    }

    // BIP: Stage 2
    r = rng.next_double();
    cum = 0.0f;
    for (int i = 0; i < NUM_STAGE2 - 1; i++) {
        cum += probs.stage2[i];
        if (r < cum) {
            return static_cast<Event>(i + 4);  // 1B=4, 2B=5, ... ROE=11
        }
    }
    return EV_ROE;  // last bucket
}

// ============================================================
// Fallback baserunning (v1 deterministic logic, simplified)
// ============================================================

static void fallback_transition(
    uint8_t base_before, uint8_t outs_before, Event event,
    FastRng& rng,
    uint8_t& base_after, uint8_t& outs_after, uint8_t& runs_scored
) {
    bool on1 = (base_before >> 0) & 1;
    bool on2 = (base_before >> 1) & 1;
    bool on3 = (base_before >> 2) & 1;
    int runs = 0;
    bool nb[3] = {false, false, false};
    int added_outs = 0;

    switch (event) {
    case EV_HR:
        runs = 1 + on1 + on2 + on3;
        break;

    case EV_3B:
        runs = on1 + on2 + on3;
        nb[2] = true;
        break;

    case EV_2B:
        if (on3) runs++;
        if (on2) runs++;
        if (on1) { if (rng.next_double() < 0.60) runs++; else nb[2] = true; }
        nb[1] = true;
        break;

    case EV_1B:
        if (on3) runs++;
        if (on2) { if (rng.next_double() < 0.59) runs++; else nb[2] = true; }
        if (on1) { if (rng.next_double() < 0.29) nb[2] = true; else nb[1] = true; }
        nb[0] = true;
        break;

    case EV_BB: case EV_HBP:
        if (on1 && on2 && on3) runs++;
        if (on1 && on2) nb[2] = true;
        if (on1) nb[1] = true;
        nb[0] = true;
        // preserve existing runners not displaced
        if (!on1 && on2) nb[1] = true;
        if (!on1 && !on2 && on3) nb[2] = true;
        if (on1 && !on2 && on3) nb[2] = true;
        break;

    case EV_K:
        added_outs = 1;
        nb[0] = on1; nb[1] = on2; nb[2] = on3;
        break;

    case EV_GO:
        added_outs = 1;
        if (on1 && outs_before < 2 && rng.next_double() < 0.12) {
            added_outs = 2;
            nb[1] = on2; nb[2] = on3;
        } else {
            if (on3 && outs_before < 2 && rng.next_double() < 0.55) runs++;
            nb[0] = false;
            nb[1] = on1 || on2;
            nb[2] = on3 && (runs == 0);
        }
        break;

    case EV_FO: case EV_LO:
        added_outs = 1;
        if (on3 && outs_before < 2 && event == EV_FO && rng.next_double() < 0.45) runs++;
        nb[0] = on1; nb[1] = on2; nb[2] = on3 && (runs == 0);
        break;

    case EV_FC:
        added_outs = 1;
        nb[0] = true;
        if (on1) nb[1] = true;
        break;

    case EV_ROE:
        nb[0] = true;
        if (on3) runs++;
        if (on2) nb[2] = true;
        if (on1) nb[1] = true;
        break;

    case EV_SAC:
        added_outs = 1;
        if (on3 && outs_before < 2) runs++;
        if (on1) nb[1] = true;
        if (on2) nb[2] = true;
        break;
    }

    base_after = (nb[0] ? 1 : 0) | (nb[1] ? 2 : 0) | (nb[2] ? 4 : 0);
    outs_after = std::min(outs_before + added_outs, 3);
    runs_scored = static_cast<uint8_t>(runs);
}

// ============================================================
// resolve_transition
// ============================================================

void resolve_transition(
    const TransitionTable& table,
    uint8_t base_before, uint8_t outs_before, Event event,
    SpeedBucket speed, DefenseBucket defense,
    float speed_fast_mod, float speed_slow_mod,
    float def_good_mod, float def_poor_mod,
    int min_obs,
    FastRng& rng,
    uint8_t& base_after, uint8_t& outs_after, uint8_t& runs_scored
) {
    const auto& cell = table.cells[base_before][outs_before][event];

    if (cell.total_obs < min_obs || cell.transitions.empty()) {
        fallback_transition(base_before, outs_before, event, rng,
                          base_after, outs_after, runs_scored);
        return;
    }

    // Copy probabilities for modification
    int n = static_cast<int>(cell.transitions.size());
    std::vector<float> probs(n);
    for (int i = 0; i < n; i++) {
        probs[i] = cell.transitions[i].probability;
    }

    // Apply speed modifier
    if (speed != SPEED_AVG) {
        float mod = (speed == SPEED_FAST) ? speed_fast_mod : speed_slow_mod;
        for (int i = 0; i < n; i++) {
            float adv_score = cell.transitions[i].runs_scored * 2.0f;
            // Count runners in base_after
            uint8_t ba = cell.transitions[i].base_after;
            adv_score += ((ba >> 0) & 1) + ((ba >> 1) & 1) + ((ba >> 2) & 1);

            if (adv_score > 1.0f) {
                probs[i] *= mod;
            } else {
                probs[i] *= (2.0f - mod);
            }
        }
    }

    // Apply defense modifier
    if (defense != DEF_AVG) {
        float mod = (defense == DEF_GOOD) ? def_good_mod : def_poor_mod;
        for (int i = 0; i < n; i++) {
            bool has_adv = cell.transitions[i].runs_scored > 0 ||
                           cell.transitions[i].base_after > 0;
            if (has_adv) {
                probs[i] *= mod;
            } else {
                probs[i] *= (2.0f - mod);
            }
        }
    }

    // Normalize
    float total = 0.0f;
    for (int i = 0; i < n; i++) total += probs[i];
    if (total <= 0.0f) {
        fallback_transition(base_before, outs_before, event, rng,
                          base_after, outs_after, runs_scored);
        return;
    }

    // Sample
    double r = rng.next_double() * total;
    float cum = 0.0f;
    int chosen = n - 1;
    for (int i = 0; i < n; i++) {
        cum += probs[i];
        if (r < cum) { chosen = i; break; }
    }

    base_after = cell.transitions[chosen].base_after;
    outs_after = cell.transitions[chosen].outs_after;
    runs_scored = cell.transitions[chosen].runs_scored;
}

// ============================================================
// simulate_half_inning
// ============================================================

int simulate_half_inning(
    const GameSetup& setup,
    const Batter* lineup,
    int& lineup_pos,
    int pitcher_idx,
    bool is_home_batting,
    DefenseBucket fielding_defense,
    bool ghost_runner,
    FastRng& rng
) {
    int runs = 0;
    int outs = 0;
    uint8_t base_state = ghost_runner ? 0b010 : 0b000; // 010 = runner on 2B

    int team = is_home_batting ? 1 : 0;

    while (outs < 3) {
        int batter_idx = lineup_pos % LINEUP_SIZE;
        const Batter& batter = lineup[batter_idx];
        const MatchupProbs& probs = setup.matchup_cache[team][batter_idx][pitcher_idx];

        Event outcome = resolve_pa(probs, rng);

        uint8_t new_base, new_outs, scored;
        resolve_transition(
            setup.transitions,
            base_state, static_cast<uint8_t>(outs), outcome,
            batter.speed, fielding_defense,
            setup.speed_fast_modifier, setup.speed_slow_modifier,
            setup.defense_good_modifier, setup.defense_poor_modifier,
            setup.min_transition_obs,
            rng,
            new_base, new_outs, scored
        );

        runs += scored;
        outs = std::min(static_cast<int>(new_outs), 3);
        base_state = new_base;
        lineup_pos++;
    }

    return runs;
}

// ============================================================
// simulate_game
// ============================================================

// Select bullpen role based on inning, score differential, and current role
int select_bullpen_role(int inning, int score_diff, int current_role) {
    // score_diff > 0 = this team leads
    // Closer: 9th+, leading by 1-3 runs
    if (inning >= 9 && score_diff >= 1 && score_diff <= 3)
        return 4; // closer

    // Bridge: 8th or 9th non-save
    if (inning >= 8)
        return 3; // bridge

    // Setup late: 7th
    if (inning >= 7)
        return 2; // setup_late

    // Setup early: 5th-6th or blowout relief
    return 1; // setup_early
}

// Check if pitcher should be pulled (pre-inning)
bool should_pull_pitcher(int pitcher_idx, int pa_faced, float pull_threshold, int inning) {
    if (pitcher_idx > 0) {
        // Already in bullpen — don't re-pull (each role pitches their stint)
        // But pull if bullpen arm has faced 12+ batters (long relief limit)
        return pa_faced >= 12;
    }
    // Starter: pull on pitch count OR TTO
    float est_pitches = pa_faced * PITCH_COUNT_PER_PA;
    if (est_pitches >= pull_threshold) return true;
    if (pa_faced >= TTO_THRESHOLD) return true;  // 3rd time through order
    return false;
}

SimResult simulate_game(const GameSetup& setup, FastRng& rng) {
    int away_score = 0, home_score = 0;
    int away_pos = 0, home_pos = 0;

    // Pitcher state: 0=starter, 1-4=bullpen roles
    int home_pitcher_idx = 0;
    int away_pitcher_idx = 0;
    int home_pa_faced = 0, away_pa_faced = 0;

    for (int inning = 1; inning <= setup.max_innings; inning++) {
        bool ghost = (setup.max_innings > 5 && inning >= MANFRED_INNING);

        // --- Top: Away batting, Home pitching ---
        // Pre-inning pull check
        if (should_pull_pitcher(home_pitcher_idx, home_pa_faced,
                                setup.fatigue_pull_threshold, inning)) {
            int score_diff = home_score - away_score;
            int new_role = select_bullpen_role(inning, score_diff, home_pitcher_idx);
            // Don't downgrade role (e.g., don't go from bridge back to setup)
            if (new_role > home_pitcher_idx || home_pitcher_idx == 0) {
                home_pitcher_idx = new_role;
                home_pa_faced = 0;
            }
        }

        int top_runs = simulate_half_inning(
            setup, setup.away_lineup, away_pos,
            home_pitcher_idx, false, setup.home_defense,
            ghost, rng
        );
        away_score += top_runs;
        home_pa_faced += 3;

        // Mid-game blowup: if 3+ runs scored this inning, pull for next
        if (top_runs >= BLOWUP_RUNS_THRESHOLD && home_pitcher_idx == 0) {
            int score_diff = home_score - away_score;
            home_pitcher_idx = select_bullpen_role(inning, score_diff, 0);
            home_pa_faced = 0;
        }

        // --- Bottom: Home batting, Away pitching ---
        if (inning >= 9 && home_score > away_score && setup.max_innings > 5) {
            break;
        }

        // Pre-inning pull check
        if (should_pull_pitcher(away_pitcher_idx, away_pa_faced,
                                setup.fatigue_pull_threshold, inning)) {
            int score_diff = away_score - home_score;
            int new_role = select_bullpen_role(inning, score_diff, away_pitcher_idx);
            if (new_role > away_pitcher_idx || away_pitcher_idx == 0) {
                away_pitcher_idx = new_role;
                away_pa_faced = 0;
            }
        }

        int bot_runs = simulate_half_inning(
            setup, setup.home_lineup, home_pos,
            away_pitcher_idx, true, setup.away_defense,
            ghost, rng
        );
        home_score += bot_runs;
        away_pa_faced += 3;

        // Mid-game blowup
        if (bot_runs >= BLOWUP_RUNS_THRESHOLD && away_pitcher_idx == 0) {
            int score_diff = away_score - home_score;
            away_pitcher_idx = select_bullpen_role(inning, score_diff, 0);
            away_pa_faced = 0;
        }

        if (inning >= 9 && away_score != home_score && setup.max_innings > 5) {
            break;
        }
    }

    return {away_score, home_score};
}

// ============================================================
// run_monte_carlo
// ============================================================

std::vector<SimResult> run_monte_carlo(const GameSetup& setup, int n_sims, uint64_t base_seed) {
    std::vector<SimResult> results(n_sims);

    for (int i = 0; i < n_sims; i++) {
        FastRng rng(base_seed + static_cast<uint64_t>(i));
        results[i] = simulate_game(setup, rng);
    }

    return results;
}

// ============================================================
// pybind11 bindings
// ============================================================

PYBIND11_MODULE(sim_core, m) {
    m.doc() = "Baseball simulation C++ core engine";

    py::class_<SimResult>(m, "SimResult")
        .def_readonly("away_score", &SimResult::away_score)
        .def_readonly("home_score", &SimResult::home_score);

    // Main entry: takes flat numpy arrays, returns list of SimResult
    m.def("run_simulation_cpp", [](
        // matchup_cache: shape [2, 9, 5, 13] = [team, batter, pitcher, stage1(5)+stage2(8)]
        py::array_t<float> matchup_cache,
        // transition_matrix: list of (base_before, outs_before, event, base_after, outs_after, runs, prob, n_obs)
        py::array_t<float> transition_data,
        py::array_t<int> transition_indices,  // [n_rows, 3] = base_before, outs_before, event
        py::array_t<int> transition_results,  // [n_rows, 3] = base_after, outs_after, runs
        py::array_t<float> transition_probs,  // [n_rows]
        py::array_t<int> transition_obs,      // [n_rows]
        // lineup speed buckets: [2, 9]
        py::array_t<int> lineup_speeds,
        // defense buckets: [2] = away, home
        py::array_t<int> defense_buckets,
        // config
        int n_sims,
        int max_innings,
        float dampening_alpha,
        int min_transition_obs,
        float speed_fast_mod, float speed_slow_mod,
        float def_good_mod, float def_poor_mod,
        float fatigue_pull_threshold,
        uint64_t base_seed
    ) {
        GameSetup setup;
        setup.max_innings = max_innings;
        setup.dampening_alpha = dampening_alpha;
        setup.min_transition_obs = min_transition_obs;
        setup.speed_fast_modifier = speed_fast_mod;
        setup.speed_slow_modifier = speed_slow_mod;
        setup.defense_good_modifier = def_good_mod;
        setup.defense_poor_modifier = def_poor_mod;
        setup.fatigue_mild_threshold = 75.0f;
        setup.fatigue_moderate_threshold = 90.0f;
        setup.fatigue_pull_threshold = fatigue_pull_threshold;
        setup.fatigue_mild_k = 0.02f;
        setup.fatigue_mild_bb = 0.02f;
        setup.fatigue_moderate_k = 0.05f;
        setup.fatigue_moderate_bb = 0.05f;

        // Parse matchup cache: [2, 9, 5, 13]
        auto mc = matchup_cache.unchecked<4>();
        for (int team = 0; team < 2; team++) {
            for (int b = 0; b < LINEUP_SIZE; b++) {
                for (int p = 0; p < (1 + NUM_BULLPEN_ROLES); p++) {
                    for (int i = 0; i < NUM_STAGE1; i++)
                        setup.matchup_cache[team][b][p].stage1[i] = mc(team, b, p, i);
                    for (int i = 0; i < NUM_STAGE2; i++)
                        setup.matchup_cache[team][b][p].stage2[i] = mc(team, b, p, NUM_STAGE1 + i);
                }
            }
        }

        // Parse transition table
        auto ti = transition_indices.unchecked<2>();
        auto tr = transition_results.unchecked<2>();
        auto tp = transition_probs.unchecked<1>();
        auto to_ = transition_obs.unchecked<1>();

        // Clear
        for (int b = 0; b < NUM_BASE_STATES; b++)
            for (int o = 0; o < NUM_OUTS; o++)
                for (int e = 0; e < NUM_EVENTS; e++) {
                    setup.transitions.cells[b][o][e].transitions.clear();
                    setup.transitions.cells[b][o][e].total_obs = 0;
                }

        int n_trans = transition_indices.shape(0);
        for (int i = 0; i < n_trans; i++) {
            int base = ti(i, 0);
            int outs = ti(i, 1);
            int event = ti(i, 2);
            if (base < 0 || base >= NUM_BASE_STATES) continue;
            if (outs < 0 || outs >= NUM_OUTS) continue;
            if (event < 0 || event >= NUM_EVENTS) continue;

            Transition t;
            t.base_after = static_cast<uint8_t>(tr(i, 0));
            t.outs_after = static_cast<uint8_t>(tr(i, 1));
            t.runs_scored = static_cast<uint8_t>(tr(i, 2));
            t.probability = tp(i);

            auto& cell = setup.transitions.cells[base][outs][event];
            cell.transitions.push_back(t);
            cell.total_obs += to_(i);
        }

        // Lineup speeds
        auto ls = lineup_speeds.unchecked<2>();
        for (int b = 0; b < LINEUP_SIZE; b++) {
            setup.away_lineup[b].id = b;
            setup.away_lineup[b].speed = static_cast<SpeedBucket>(ls(0, b));
            setup.home_lineup[b].id = b;
            setup.home_lineup[b].speed = static_cast<SpeedBucket>(ls(1, b));
        }

        // Defense
        auto db = defense_buckets.unchecked<1>();
        setup.away_defense = static_cast<DefenseBucket>(db(0));
        setup.home_defense = static_cast<DefenseBucket>(db(1));

        // Run
        auto results = run_monte_carlo(setup, n_sims, base_seed);

        // Convert to numpy
        py::array_t<int> scores({n_sims, 2});
        auto s = scores.mutable_unchecked<2>();
        for (int i = 0; i < n_sims; i++) {
            s(i, 0) = results[i].away_score;
            s(i, 1) = results[i].home_score;
        }
        return scores;
    },
    "Run Monte Carlo simulation",
    py::arg("matchup_cache"),
    py::arg("transition_data"),
    py::arg("transition_indices"),
    py::arg("transition_results"),
    py::arg("transition_probs"),
    py::arg("transition_obs"),
    py::arg("lineup_speeds"),
    py::arg("defense_buckets"),
    py::arg("n_sims"),
    py::arg("max_innings") = 15,
    py::arg("dampening_alpha") = 0.65f,
    py::arg("min_transition_obs") = 30,
    py::arg("speed_fast_mod") = 1.10f,
    py::arg("speed_slow_mod") = 0.90f,
    py::arg("def_good_mod") = 0.95f,
    py::arg("def_poor_mod") = 1.05f,
    py::arg("fatigue_pull_threshold") = 100.0f,
    py::arg("base_seed") = 42
    );
}

} // namespace sim
