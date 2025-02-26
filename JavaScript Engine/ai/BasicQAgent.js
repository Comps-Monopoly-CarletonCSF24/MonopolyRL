import fs from 'fs';

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
    this.loadQTableFromFile = function(filePath){
        const data = fs.readFileSync(filePath, 'utf8');
        const lines = data.split('\n');
        let currentState = null;

        lines.forEach(line => {
            line = line.trim();
            if (line.startsWith("State:")){
                currentState = line.match(/\(([^)]+)\)/)[1];
            }else if (line.startsWith("Action:") && currentState){
                const action = line.split(':')[1];
                const qValueLine = lines[lines.indexOf(line) + 1];
                const qValue = parseFloat(qValueLine.split(':')[1]);
                this.qTable[`${currentState}:${action}`] = qValue;
            }
        });
    };
}   

// agent initialization
const agent = new BasicQAgent();
agent.loadQTableFromFile('path/to/qtable_Basic Q agent 1.txt');
