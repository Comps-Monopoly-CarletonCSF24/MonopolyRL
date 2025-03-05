import { State } from "./state.js";
import { Actions, Total_Actions, Action} from "./action.js"

export async function runModel(inputArray) {
    const inputTensor = new ort.Tensor('float32', new Float32Array(inputArray), [24]);
    const feeds = { "state_action_input": inputTensor };

    const results = await session.run(feeds);
    const output = results.q_value_output.data[0]; // Adjust based on actual model output

    return output;
}

function createModelInput(state, action){
    return [...state.state, action.action_index/Total_Actions];
}

export async function chooseAction() {
    let state = new State();
    let action_q_values = [];
    
    // Calculate Q-values for each action
    for (let i = 0; i < Total_Actions; i++) {
        let action = new Action(Actions[i]);
        let model_input = createModelInput(state, action);
        let q_value = await runModel(model_input);
        action_q_values.push(q_value);
    }
    
    // Find the maximum Q-value
    const maxQValue = Math.max(...action_q_values);
    
    // Find all indices with the maximum Q-value
    const maxIndices = action_q_values.reduce((indices, qValue, index) => {
        if (qValue === maxQValue) {
            indices.push(index);
        }
        return indices;
    }, []);
    
    // Randomly select one of the indices with the maximum Q-value
    const selectedIndex = maxIndices[Math.floor(Math.random() * maxIndices.length)];
    
    return Actions[selectedIndex];
}