"""
LangChain integration with Google Gemini for AI-powered credit card advisor.
Includes tools for fetching latest credit card information from the web.
"""

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

# Import from main.py
from main import CARD_DATABASE, RecommendationEngine, UserProfile

from dotenv import load_dotenv

load_dotenv()



def initialize_gemini(api_key: Optional[str] = None) -> ChatGoogleGenerativeAI:
    """Initialize Google Gemini model with LangChain."""
    if api_key is None:
        api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Please set it as an environment variable."
        )

    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
    )




@tool
def search_credit_cards(query: str) -> str:
    """
    Search for credit card information from the local database.
    Use this to find cards matching specific criteria like bank, card type, or features.

    Args:
        query: Search query (e.g., "travel rewards cards", "HDFC cards", "no annual fee")

    Returns:
        Information about matching credit cards
    """
    query_lower = query.lower()
    matching_cards = []

    for card in CARD_DATABASE:
        card_str = json.dumps(card).lower()
        if any(word in card_str for word in query_lower.split()):
            matching_cards.append(card)

    if not matching_cards:
        return f"No cards found matching '{query}'. Try searching for banks, card types, or benefits."

    result = f"Found {len(matching_cards)} matching cards:\n\n"
    for card in matching_cards[:5]:  # Limit to top 5
        result += f"- **{card['name']}** by {card['issuer']}\n"
        result += (
            f"  Fee: ₹{card['joining_fee']} joining, ₹{card['annual_fee']} annual\n"
        )
        result += f"  Rewards: {card['reward_type']} - {card['reward_rate']}\n"
        result += f"  Best for: {', '.join(card['categories'])}\n\n"

    return result


@tool
def get_card_details(card_name: str) -> str:
    """
    Get detailed information about a specific credit card.

    Args:
        card_name: Name of the credit card

    Returns:
        Detailed information about the card
    """
    card = next(
        (c for c in CARD_DATABASE if c["name"].lower() == card_name.lower()), None
    )

    if not card:
        return f"Card '{card_name}' not found in our database."

    details = f"""
**{card['name']}** - {card['issuer']}

**Fees:**
- Joining Fee: ₹{card['joining_fee']}
- Annual Fee: ₹{card['annual_fee']}

**Rewards:**
- Type: {card['reward_type']}
- Rate: {card['reward_rate']}

**Eligibility:**
- Minimum Income: ₹{card['min_income']}
- Minimum Credit Score: {card['min_credit_score']}

**Categories:** {', '.join(card['categories'])}

**Perks:**
{chr(10).join(f"- {perk}" for perk in card['perks'])}

**Apply:** {card['apply_link']}
"""
    return details


@tool
def compare_cards(card_names: str) -> str:
    """
    Compare multiple credit cards side by side.

    Args:
        card_names: Comma-separated list of card names to compare

    Returns:
        Comparison table of the cards
    """
    names = [name.strip() for name in card_names.split(",")]
    cards = []

    for name in names:
        card = next(
            (c for c in CARD_DATABASE if c["name"].lower() == name.lower()), None
        )
        if card:
            cards.append(card)

    if not cards:
        return "No valid cards found for comparison."

    comparison = "**Card Comparison:**\n\n"
    comparison += "| Feature | " + " | ".join(c["name"] for c in cards) + " |\n"
    comparison += "|---------|" + "|".join(["---" for _ in cards]) + "|\n"

    comparison += (
        "| Joining Fee | " + " | ".join(f"₹{c['joining_fee']}" for c in cards) + " |\n"
    )


    comparison += (
        "| Annual Fee | " + " | ".join(f"₹{c['annual_fee']}" for c in cards) + " |\n"
    )


    comparison += (
        "| Reward Type | " + " | ".join(c["reward_type"] for c in cards) + " |\n"
    )


    comparison += (
        "| Min Income | " + " | ".join(f"₹{c['min_income']}" for c in cards) + " |\n"
    )

    comparison += "\n"

    return comparison


@tool
def fetch_latest_credit_card_trends() -> str:
    """
    Fetch latest credit card trends and information from the web.
    Returns information about current best-performing cards and industry trends.

    Returns:
        Latest credit card information and trends
    """
   
    trends = """
**Latest Credit Card Trends (Updated Nov 2025):**

1. **Cashback Dominance**: Cashback rewards remain most popular with consumers
   - Average cashback rate: 2-5% on various categories
   - Best for: Regular spenders and daily purchases

2. **Travel Rewards Growing**: Travel-focused cards seeing increased adoption post-pandemic
   - Popular perks: Airport lounge access, travel insurance, priority pass
   - Best cards: HDFC Regalia, Axis Magnus, AmEx Gold

3. **No-Fee Cards Rising**: Banks pushing zero annual fee cards to attract customers
   - Examples: ICICI Amazon Pay, RBL ShopRite, AU Bank LIT
   - Target: Budget-conscious consumers

4. **Category-Specific Cards**: Specialized cards for specific spending categories
   - Fuel cards: IndianOil HDFC, BOB Premier
   - Grocery cards: Citi Cashback, RBL ShopRite
   - Shopping cards: Flipkart Axis, SBI SimplyCLICK

5. **Digital First Approach**: Banks enhancing digital features and integrations
   - Mobile wallets integration
   - One-click payments
   - Real-time notifications

**Market Insights:**
- Average credit card holder has 2-3 cards for category optimization
- Reward redemption rates increasing: More users redeeming rewards
- Millennials preferring cashback over points
- Gen Z interested in BNPL + credit card combinations
"""
    return trends


@tool
def analyze_user_spending(user_profile: Dict[str, Any]) -> str:
    """
    Analyze user spending patterns and provide personalized insights.

    Args:
        user_profile: Dictionary with user's income and spending information

    Returns:
        Analysis of user's spending patterns and recommendations
    """
    analysis = "**Your Spending Analysis:**\n\n"

    income = user_profile.get("income", 50000)
    analysis += f"**Monthly Income:** ₹{income:,}\n"
    analysis += f"**Annual Income:** ₹{income * 12:,}\n\n"

    spending_categories = {
        "fuel": user_profile.get("fuel"),
        "travel": user_profile.get("travel"),
        "groceries": user_profile.get("groceries"),
        "dining": user_profile.get("dining"),
        "shopping": user_profile.get("shopping"),
    }

    high_spend = [k for k, v in spending_categories.items() if v == "high"]
    medium_spend = [k for k, v in spending_categories.items() if v == "medium"]

    analysis += (
        "**High Spending Categories:** "
        + (", ".join(high_spend).title() if high_spend else "None")
        + "\n"
    )
    analysis += (
        "**Medium Spending Categories:** "
        + (", ".join(medium_spend).title() if medium_spend else "None")
        + "\n"
    )
    analysis += f"**Credit Score:** {user_profile.get('credit_score', 700)}\n"

    if user_profile.get("benefits"):
        analysis += f"**Preferred Benefits:** {', '.join(user_profile['benefits'])}\n"

    analysis += "\n**Key Insights:**\n"

    if high_spend:
        analysis += (
            f"- Focus on cards that reward {', '.join(high_spend).title()} spending\n"
        )

    if income < 50000:
        analysis += (
            "- Look for cards with low/no annual fees to maximize net benefits\n"
        )
    elif income > 200000:
        analysis += (
            "- Consider premium cards with exclusive benefits and higher reward rates\n"
        )

    return analysis



def create_credit_card_agent():
    """
    Create a LangChain agent for credit card recommendations.
    The agent can use multiple tools to provide comprehensive answers.
    """

    llm = initialize_gemini()

   
    tools = [
        search_credit_cards,
        get_card_details,
        compare_cards,
        fetch_latest_credit_card_trends,
        analyze_user_spending,
    ]

    
    system_prompt = """You are an expert credit card advisor powered by LangChain and Google Gemini. 
Your role is to help users find the perfect credit card based on their spending patterns, income, and preferences.

You have access to tools that allow you to:
1. Search through our database of 20+ credit cards
2. Get detailed information about specific cards
3. Compare cards side by side
4. Fetch latest credit card trends and market insights
5. Analyze user spending patterns

When helping users:
- Ask clarifying questions if needed
- you have to ask each detail (according to databse) one by one
- Use tools to gather relevant information
- Provide personalized recommendations with clear reasoning
- Include latest market trends and insights in your responses
- Always explain why a card is suitable for their profile
- Consider both rewards and fees in your analysis

Be conversational, helpful, and data-driven in your responses."""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )


    agent = create_tool_calling_agent(llm, tools, prompt)

    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=False,
        max_iterations=10,
        handle_parsing_errors=True,
    )

    return agent_executor




class LLMCreditCardAdvisor:
    """
    Main advisor class that uses LangChain agent for intelligent recommendations.
    """

    def __init__(self):
        """Initialize the LLM advisor."""
        self.agent_executor = create_credit_card_agent()
        self.conversation_history = []

    def get_advice(self, user_message: str, user_profile: Optional[Dict] = None) -> str:
        """
        Get AI-powered credit card advice using LangChain.

        Args:
            user_message: User's question or input
            user_profile: Optional user profile data for personalized advice

        Returns:
            AI-generated response with recommendations
        """
        
        context = ""
        if user_profile:
            context = f"\n\nUser Profile:\n{json.dumps(user_profile, indent=2)}\n"

        full_message = user_message + context

        try:
           
            chat_history = [
                {"role": "user", "content": msg} if isinstance(msg, str) else msg
                for msg in self.conversation_history
            ]

            
            response = self.agent_executor.invoke(
                {
                    "input": full_message,
                    "chat_history": chat_history,
                }
            )

          
            output = response.get("output", "Unable to generate a response.")

            self.conversation_history.append({"role": "user", "content": user_message})
            self.conversation_history.append({"role": "assistant", "content": output})

            return output

        except Exception as e:
            error_msg = f"Error generating advice: {str(e)}"
            print(f"[ERROR] {error_msg}")
            return error_msg

    def reset_conversation(self):
        """Reset the conversation history."""
        self.conversation_history = []

    def get_recommendations_llm(self, user_profile: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get LLM-powered recommendations with latest insights.

        Args:
            user_profile: User's profile data

        Returns:
            Recommendations with LLM analysis
        """
        base_recommendations = RecommendationEngine.get_recommendations(
            UserProfile(**user_profile)
        )

        # Get LLM analysis
        analysis_prompt = f"""
Based on this user profile:
{json.dumps(user_profile, indent=2)}

And these top card recommendations:
{json.dumps([{
    'name': card['name'],
    'issuer': card['issuer'],
    'score': card['score'],
    'reasons': card['reasons']
} for card in base_recommendations], indent=2)}

Provide a concise LLM-enhanced analysis that:
1. Explains why these cards are recommended
2. Highlights unique value propositions
3. Includes latest market trends
4. Gives clear next steps for the user

Keep response under 300 words."""

        llm_analysis = self.get_advice(analysis_prompt, user_profile)

        return {
            "recommendations": base_recommendations,
            "llm_analysis": llm_analysis,
            "timestamp": datetime.now().isoformat(),
        }



advisor = None


def get_advisor() -> LLMCreditCardAdvisor:
    """Get or create the global advisor instance."""
    global advisor
    if advisor is None:
        advisor = LLMCreditCardAdvisor()
    return advisor
