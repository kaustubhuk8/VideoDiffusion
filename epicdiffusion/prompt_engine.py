from typing import List, Dict, Optional
from dataclasses import dataclass
import openai  # Will be optional if no API key provided

@dataclass
class StoryElements:
    characters: List[str]
    setting: str
    actions: List[str]
    style: Optional[str] = None

class PromptRefiner:
    def __init__(self, llm_api_key: Optional[str] = None):
        """Initialize prompt refiner with optional LLM support."""
        self.llm_enabled = bool(llm_api_key)
        if llm_api_key:
            openai.api_key = llm_api_key

    def decompose_prompt(self, prompt: str) -> StoryElements:
        """Break down prompt into narrative elements using Chain of Thought."""
        if self.llm_enabled:
            return self._decompose_with_llm(prompt)
        return self._decompose_basic(prompt)

    def _decompose_with_llm(self, prompt: str) -> StoryElements:
        """Use LLM for advanced prompt decomposition."""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Decompose this story prompt into characters, setting, and actions."
            }, {
                "role": "user",
                "content": prompt
            }]
        )
        return self._parse_llm_response(response.choices[0].message.content)

    def _decompose_basic(self, prompt: str) -> StoryElements:
        """Basic decomposition without LLM."""
        # Simple fallback implementation
        return StoryElements(
            characters=["main character"],
            setting="generic setting",
            actions=["main action"]
        )

    def refine_prompt(self, elements: StoryElements) -> str:
        """Reconstruct prompt from decomposed elements with enhancements."""
        characters = ", ".join(elements.characters)
        actions = ", ".join(elements.actions)
        return f"{characters} in {elements.setting}, {actions}"

    def _parse_llm_response(self, text: str) -> StoryElements:
        """Parse LLM response into structured elements."""
        # Implementation would parse the LLM's response
        # This is a simplified version
        return StoryElements(
            characters=["parsed character"],
            setting="parsed setting",
            actions=["parsed action"]
        )
