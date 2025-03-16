# Explinations

## Test 0

The goal of this test is to explore potential ideas for prompting the models. We’re not focused on full model accuracy at this stage, as we assume that if a quantized model performs well with a given prompt, the FP16 version will perform similarly. To save time and, especially, the cost of renting the GPU, we use the default Ollama models and their quantizations. Most models run in Q4-K-M, though Q8 or others may also be used, depending on what Ollama selects based on the model and its size. We avoid FP16 computation here due to its higher resource requirements, and larger models naturally take longer to process. As a result, we focus on quantized variants of the models to speed up the process.

Quantized models have shown to perform well in our initial tests, achieving around 85% accuracy. The main downside is that smaller models tend to suffer more from quantization loss compared to larger models, which have more parameters. To balance time and cost, we use Q4 and similar quantizations in this test, and the results will help inform further experiments with less-quantized models that still fit within the 24 GB of RAM on the RTX 3090 / 4090.
