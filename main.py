import time
from dotenv import load_dotenv
load_dotenv()
from batch_router.core.base import BatchStatus, ProviderId
from batch_router.providers import GoogleGenAIProvider
from batch_router.core.input import InputBatch, InputRequest, InputMessage, InputMessageRole
from batch_router.core.base import TextContent, InferenceParams

provider = GoogleGenAIProvider() # add api key to .env file

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
input_batch = InputBatch(
    requests=requests
)

google_params = InferenceParams(
    model_id="gemini-2.5-flash-lite",
    provider_id=ProviderId.GOOGLE,
    system_prompt="Answer like a pirate.",
    max_output_tokens=128
)

google_genai_batch = input_batch.with_params(google_params)

print(google_genai_batch)

# You can also use the same input batch for multiple providers by configuring it with different models and providers.
# openai_batch = input_batch.with_config(
#     config=BatchConfig(
#         provider_id=ProviderId.OPENAI,
#         model_id="gpt-5-mini"
#     )
# )

google_genai_batch_id = provider.send_batch(google_genai_batch)

total_tokens = provider.count_input_batch_tokens(google_genai_batch)

print(f"This test batch has a total of {total_tokens} tokens.")

while provider.poll_status(google_genai_batch_id) != BatchStatus.COMPLETED:
    time.sleep(5)
    print(f"Batch {google_genai_batch_id} is {provider.poll_status(google_genai_batch_id)}")

google_genai_batch = provider.get_results(google_genai_batch_id)
for request in google_genai_batch.requests:
    print(f"Request {request.custom_id} has text {request.messages[0].contents[0].text}")
