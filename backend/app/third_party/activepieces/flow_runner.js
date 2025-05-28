// flow_runner.js

const fs = require("fs");

async function run() {
  try {
    const inputPath = process.argv[2]; // path to flow definition
    const inputsRaw = process.argv[3]; // stringified JSON

    const flowDefinition = require(inputPath); // loads .js/.json flow
    const inputs = JSON.parse(inputsRaw);

    // You'd replace this with actual Activepieces engine logic
    // For now we just simulate output
    console.log(JSON.stringify({
      status: "success",
      message: `Ran flow at ${inputPath} with inputs`,
      inputs
    }));
  } catch (err) {
    console.error(JSON.stringify({
      status: "error",
      message: err.message
    }));
    process.exit(1);
  }
}

run();
