// interface that defines what an MDP is
class MDP{
    constructor(state, action){
        this.state = state;
        this.action = action;
    }
    get_states(){   // Return all states of this MDP
        return;
    }
    get_actions(state){ // Return all actions with non-zero probability from this state
        return;
    }
    get_transitions(state,action){
        /* Return all non-zero probablity transitions for this action 
        from this state, as a list of (state, probability) parirs */
        return;
    }
    get_reward(state,action,next_state){
        // Return the reward for transitioning from state to next_state via action
        return;
    }
    is_terminal(state){ //Return true iff state is a terminal state of this MDP
        return false;
    }
    get_initial_state(){    // Return the initial state of this MDP
        return;
    }
    get_discount_factor(){  // Return the discount factor for this MDP
        return;
    }
    get_goal_states(){  //Return all goal states of this MDP
        return;
    }
}
