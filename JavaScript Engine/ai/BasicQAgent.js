export function BasicQAgent(p) { // match the existing AI interface that game.js is expecting
    this.alertList = "";
    // q learning variables
    this.qTable = {};
    this.actions = [0, 1]; // define in constructor so no need to pass during intilization
    this.alpha = 0.2; // Learning rate
    this.gamma = 0.95; // Discount factor
    this.epsilon = 0.4; // Exploration rate

    // Required interface methods that game.js uses
    this.buyProperty = function(index) {
        // TODO:Q -learning logic for buying
    };

    this.acceptTrade = function(trade) {
        // TODO:Q -learning logic for accepting
    };

    this.makeTrade = function(trade) {
        // TODO: Q -learning logic for making
    };

    this.getQValue = function(state, action) {
        return this.qTable[`${state}:${action}`] || 0.0;
    };
    
    this.updateQValue = function(state, action, reward, nextState) {
        const bestNextQ = Math.max(...this.actions.map(a => this.getQValue(nextState, a)));
        this.qTable[`${state}:${action}`] = this.getQValue(state, action) +
            this.alpha * (reward + this.gamma * bestNextQ - this.getQValue(state, action));
    };

    this.chooseAction= function(state){
        // if (Math.random() < this.epsilon) {
        //     return this.actions[Math.floor(Math.random() * this.actions.length)]; // Explore
        // } else {
            return this.actions.reduce((bestAction, a) => 
                this.getQValue(state, a) > this.getQValue(state, bestAction) ? a : bestAction
            ); // Exploit
        
    };
}

// agent initialization
const agent = new BasicQAgent();
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
