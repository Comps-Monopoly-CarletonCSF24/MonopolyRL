import * as ort from 'onnxruntime-web';
const session = await ort.InferenceSession.create('./onnx_model.onnx');

export async function runModel(inputArray) {
    const inputTensor = new ort.Tensor('float32', new Float32Array(inputArray), [24]);
    const feeds = { "state_action_input": inputTensor };

    const results = await session.run(feeds);

    const output = results.q_value_output.cpuData[0];
    console.log('Model Output:', output);
    return output
}

// for dummy data
const testInput = Array(24).fill(0); // replace this part with the actual data
runModel(testInput);