import { State } from "./state.js";
import { Actions, Total_Actions, Action} from "./action.js"

export async function runModel(inputArray) {
    console.log("testss");
    console.log(inputArray.length);

    const inputTensor = new ort.Tensor('float32', new Float32Array(inputArray), [24]);
    const feeds = { "state_action_input": inputTensor };

    const results = await session.run(feeds);
    const output = results.q_value_output.data[0]; // Adjust based on actual model output

    console.log('Model Output:', output);
    document.getElementById("output").textContent = "Model Output: " + output;
    return output;
}

function createModelInput(state, action){
    return new Array(24).fill(0);
    //[...state.state, action.action_index/Total_Actions];
}

export async function chooseAction(){
    let state = new State();
    let action_q_values = [];
    for(let i = 0 ; i < 3; i++){
        let action = new Action(Actions[i]);
        let model_input = createModelInput(state, action);
        let q_value = await runModel(model_input)
        console.log("input" + model_input.length)
        action_q_values.push(q_value)
    }
    return action_q_values
}