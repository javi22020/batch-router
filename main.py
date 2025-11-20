"""
This script demonstrates the usage of the Batch Router library for batch inference.

It initializes a provider (Google GenAI in this example), creates a batch of input requests,
configures them with inference parameters, sends the batch for processing, polls for completion,
and retrieves the results.
"""

import time
from dotenv import load_dotenv
load_dotenv()
from batch_router.core.base import BatchStatus, ProviderId
from batch_router.providers import GoogleGenAIProvider
from batch_router.core.input import InputBatch, InputRequest, InputMessage, InputMessageRole
from batch_router.core.base import TextContent, InferenceParams

# Initialize the provider (ensure GOOGLE_API_KEY is set in .env)
provider = GoogleGenAIProvider()

# Create a list of input requests
requests = [
    InputRequest(
        custom_id=f"request_{i+1}",
        messages=[
            InputMessage(
                role=InputMessageRole.USER,
                contents=[
                    TextContent(text="Hello, how are you?")
                ]
            )
        ]
    )
    for i in range(10)
]

# Create the input batch object
input_batch = InputBatch(
    requests=requests
)

# Define inference parameters
google_params = InferenceParams(
    model_id="gemini-2.5-flash-lite",
    provider_id=ProviderId.GOOGLE,
    system_prompt="Answer like a pirate.",
    max_output_tokens=128
)

# Apply parameters to the batch
google_genai_batch = input_batch.with_params(google_params)

print(google_genai_batch)

# Example of reusing the batch for another provider (commented out)
# openai_batch = input_batch.with_config(
#     config=BatchConfig(
#         provider_id=ProviderId.OPENAI,
#         model_id="gpt-5-mini"
#     )
# )

# Send the batch to the provider
google_genai_batch_id = provider.send_batch(google_genai_batch)

# Count tokens (optional)
total_tokens = provider.count_input_batch_tokens(google_genai_batch)
print(f"This test batch has a total of {total_tokens} tokens.")

# Poll for completion
while provider.poll_status(google_genai_batch_id) != BatchStatus.COMPLETED:
    time.sleep(5)
    print(f"Batch {google_genai_batch_id} is {provider.poll_status(google_genai_batch_id)}")

# Retrieve results
google_genai_batch = provider.get_results(google_genai_batch_id)
for request in google_genai_batch.requests:
    print(request)
