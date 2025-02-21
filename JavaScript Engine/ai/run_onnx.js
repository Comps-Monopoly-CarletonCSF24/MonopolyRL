const session = new onnx.InferenceSession()
await session.loadModel('./onnx_model.onnx');

export async function runModel(inputArray) {
    const inputTensor = new onnx.Tensor('float32', new Float32Array(inputArray), [24]);
    const feeds = { "state_action_input": inputTensor };

    const results = await session.run(feeds);

    const output = results.q_value_output.cpuData[0];
    console.log('Model Output:', output);
    return output
}

function createModelInput(state, action){
    return [...state.state, action.action_index/Total_Actions];
}

export async function chooseAction(){
    let state = new State();
    action_q_values = [];
    for(i = 0 ; i < Total_Actions; i++){
        action = new Action(Actions[i]);
        model_input = createModelInput(state, action);
        q_value = runModel(model_input)
        action_q_values.push(q_value)
    }
    return action_q_values
}