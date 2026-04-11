import anthropic
from dotenv import load_dotenv

load_dotenv()  # reads your .env file

client = anthropic.Anthropic()  # automatically picks up ANTHROPIC_API_KEY

message = client.messages.create(
    model="claude-opus-4-5",
    max_tokens=100,
    messages=[
        {"role": "user", "content": "Reply with just the words: connection successful"}
    ]
)

print(message.content[0].text)