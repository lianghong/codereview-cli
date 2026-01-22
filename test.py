#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : test.py
# Author            : Lianghong Fei <feilianghong@gmail.com>
# Date              : 2025-11-16
# Last Modified Date: 2025-11-16
# Last Modified By  : Lianghong Fei <feilianghong@gmail.com>
# pip install -qU "langchain[anthropic]" to call the model

from langchain.agents import create_agent
from langchain_aws import ChatBedrockConverse
from dataclasses import dataclass

model_id = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
aws_region = "us-west-2"
max_tokens = 4096
temperature=0.5

@dataclass
class ResponseFormat:
    """Response schema for the agent."""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None


SYSTEM_PROMPT = """You are an expert weather forecaster, who speaks in puns.

You have access to two tools:

- get_weather_for_location: use this to get the weather for a specific location
- get_user_location: use this to get the user's location

If a user asks you for the weather, make sure you know the location. If you can tell from the question that they mean wherever they are, use the get_user_location tool to find their location."""


def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"


agent = create_agent(
    model=ChatBedrockConverse(
        model=model_id,
        region_name=aws_region
    ),
    tools=[get_weather],
    system_prompt=SYSTEM_PROMPT,
    response_format=ResponseFormat,
)

# Run the agent
response = agent.invoke(
    {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
)
print(response['structured_response'])
#print(response)

