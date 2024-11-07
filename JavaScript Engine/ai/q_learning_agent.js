class QLearningAgent {
    constructor(actions) {
        this.qTable = {};
        this.actions = actions;
        this.alpha = 0.2; // Learning rate
        this.gamma = 0.95; // Discount factor
        // this.epsilon = 1.0; // Exploration rate
    }

    getQValue(state, action) {
        return this.qTable[`${state}:${action}`] || 0.0;
    }

    updateQValue(state, action, reward, nextState) {
        const bestNextQ = Math.max(...this.actions.map(a => this.getQValue(nextState, a)));
        this.qTable[`${state}:${action}`] = this.getQValue(state, action) +
            this.alpha * (reward + this.gamma * bestNextQ - this.getQValue(state, action));
    }

    chooseAction(state) {
        // if (Math.random() < this.epsilon) {
        //     return this.actions[Math.floor(Math.random() * this.actions.length)]; // Explore
        // } else {
            return this.actions.reduce((bestAction, a) => 
                this.getQValue(state, a) > this.getQValue(state, bestAction) ? a : bestAction
            ); // Exploit
        
    }
}

// agent initialization
const agent = new QLearningAgent([0, 1]); // 0 = do nothing, 1 = buy property

// Training loop
// for (let episode = 0; episode < 1000; episode++) { // Run multiple episodes
//     let state = game.reset(); // the ai does not know that the game is yet... we have to get it to learn that
//     let done = false;
//     while (!done) {
//         const action = agent.chooseAction(state);
//         const [nextState, reward] = game.step(action); // also have to define state so that we define rewards
//         agent.updateQValue(state, action, reward, nextState);
//         state = nextState;
//     }
// }
