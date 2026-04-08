#pragma once
#include <cstdint>
#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <cmath>

namespace sim {

// ============================================================
// Constants
// ============================================================

constexpr int NUM_BASE_STATES = 8;   // 000~111
constexpr int NUM_OUTS = 3;          // 0,1,2
constexpr int NUM_EVENTS = 13;       // K,BB,HBP,HR,1B,2B,3B,GO,FO,LO,FC,ROE,SAC
constexpr int NUM_STAGE1 = 5;        // K,BB,HBP,HR,BIP
constexpr int NUM_STAGE2 = 8;        // 1B,2B,3B,GO,FO,LO,FC,ROE
constexpr int LINEUP_SIZE = 9;
constexpr int NUM_BULLPEN_ROLES = 4; // setup_early, setup_late, bridge, closer
constexpr int MAX_INNINGS = 15;
constexpr int MANFRED_INNING = 10;
constexpr float PITCH_COUNT_PER_PA = 4.0f;
constexpr int TTO_THRESHOLD = 27;           // 3x through order -> pull check
constexpr int BLOWUP_RUNS_THRESHOLD = 3;    // 3+ runs in half-inning -> pull

// Event type indices
enum Event : uint8_t {
    EV_K = 0, EV_BB = 1, EV_HBP = 2, EV_HR = 3,
    EV_1B = 4, EV_2B = 5, EV_3B = 6,
    EV_GO = 7, EV_FO = 8, EV_LO = 9,
    EV_FC = 10, EV_ROE = 11, EV_SAC = 12
};

// Speed bucket
enum SpeedBucket : uint8_t { SPEED_SLOW = 0, SPEED_AVG = 1, SPEED_FAST = 2 };

// Defense bucket
enum DefenseBucket : uint8_t { DEF_POOR = 0, DEF_AVG = 1, DEF_GOOD = 2 };

// ============================================================
// Data Structures
// ============================================================

struct MatchupProbs {
    float stage1[NUM_STAGE1];  // K, BB, HBP, HR, BIP
    float stage2[NUM_STAGE2];  // 1B, 2B, 3B, GO, FO, LO, FC, ROE
};

struct Transition {
    uint8_t base_after;
    uint8_t outs_after;
    uint8_t runs_scored;
    float probability;
};

struct TransitionCell {
    std::vector<Transition> transitions;
    int total_obs;
};

struct TransitionTable {
    TransitionCell cells[NUM_BASE_STATES][NUM_OUTS][NUM_EVENTS];
};

struct Batter {
    int id;
    SpeedBucket speed;
};

struct PitcherState {
    int pa_faced;
    float k_penalty;   // accumulated fatigue penalty
    float bb_penalty;
};

struct GameSetup {
    // matchup_cache[team][batter_idx][pitcher_idx]
    // team: 0=away, 1=home
    // pitcher_idx: 0=opp_starter, 1-4=opp_bullpen roles
    MatchupProbs matchup_cache[2][LINEUP_SIZE][1 + NUM_BULLPEN_ROLES];

    TransitionTable transitions;

    Batter away_lineup[LINEUP_SIZE];
    Batter home_lineup[LINEUP_SIZE];

    DefenseBucket away_defense;
    DefenseBucket home_defense;

    int max_innings;
    float dampening_alpha;

    // Fatigue thresholds (estimated pitch counts)
    float fatigue_mild_threshold;    // 75
    float fatigue_moderate_threshold; // 90
    float fatigue_pull_threshold;    // 100
    float fatigue_mild_k;            // 0.02
    float fatigue_mild_bb;           // 0.02
    float fatigue_moderate_k;        // 0.05
    float fatigue_moderate_bb;       // 0.05

    // Speed/defense modifiers
    float speed_fast_modifier;   // 1.10
    float speed_slow_modifier;   // 0.90
    float defense_good_modifier; // 0.95
    float defense_poor_modifier; // 1.05

    int min_transition_obs;  // 30
};

struct SimResult {
    int away_score;
    int home_score;
};

// ============================================================
// Core Functions
// ============================================================

// RNG: simple xoshiro256+ for speed
class FastRng {
public:
    explicit FastRng(uint64_t seed);
    double next_double();  // [0, 1)
    uint32_t next_uint32();

private:
    uint64_t s[4];
    static uint64_t splitmix64(uint64_t& x);
};

// Resolve a single plate appearance
Event resolve_pa(const MatchupProbs& probs, FastRng& rng);

// Resolve base-out transition
void resolve_transition(
    const TransitionTable& table,
    uint8_t base_before, uint8_t outs_before, Event event,
    SpeedBucket speed, DefenseBucket defense,
    float speed_fast_mod, float speed_slow_mod,
    float def_good_mod, float def_poor_mod,
    int min_obs,
    FastRng& rng,
    uint8_t& base_after, uint8_t& outs_after, uint8_t& runs_scored
);

// Simulate one half-inning
int simulate_half_inning(
    const GameSetup& setup,
    const Batter* lineup,
    int& lineup_pos,
    int pitcher_idx,    // index into matchup_cache
    bool is_home_batting,
    DefenseBucket fielding_defense,
    bool ghost_runner,
    FastRng& rng
);

// Simulate full game
SimResult simulate_game(const GameSetup& setup, FastRng& rng);

// Run N simulations
std::vector<SimResult> run_monte_carlo(const GameSetup& setup, int n_sims, uint64_t base_seed);

} // namespace sim
