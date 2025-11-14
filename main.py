import json
from datetime import datetime
from typing import Dict, List, Optional
from venv import logger

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="Credit Card Advisor API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Data Templates
class UserProfile(BaseModel):
    income: Optional[int] = None
    fuel: Optional[str] = None
    travel: Optional[str] = None
    groceries: Optional[str] = None
    dining: Optional[str] = None
    shopping: Optional[str] = None
    benefits: Optional[List[str]] = None
    credit_score: Optional[int] = None


class ChatMessage(BaseModel):
    message: str
    conversation_state: str
    user_data: UserProfile


class CreditCard(BaseModel):
    id: int
    name: str
    issuer: str
    joining_fee: int
    annual_fee: int
    reward_type: str
    reward_rate: str
    min_income: int
    min_credit_score: int
    perks: List[str]
    categories: List[str]
    image_url: str
    apply_link: str


# Credit Card Database
CARD_DATABASE = [
    {
        "id": 1,
        "name": "HDFC Regalia",
        "issuer": "HDFC Bank",
        "joining_fee": 2500,
        "annual_fee": 2500,
        "reward_type": "Reward Points",
        "reward_rate": "4 points per Rs.150",
        "min_income": 180000,
        "min_credit_score": 750,
        "perks": [
            "Airport Lounge Access",
            "Travel Insurance",
            "Dining Rewards",
            "Golf Program",
        ],
        "categories": ["travel", "dining", "luxury"],
        "image_url": "https://example.com/hdfc-regalia.jpg",
        "apply_link": "https://example.com/apply/hdfc-regalia",
    },
    {
        "id": 2,
        "name": "SBI SimplyCLICK",
        "issuer": "SBI Card",
        "joining_fee": 499,
        "annual_fee": 499,
        "reward_type": "Cashback",
        "reward_rate": "10x on online shopping",
        "min_income": 30000,
        "min_credit_score": 700,
        "perks": ["Amazon Vouchers", "Online Shopping Rewards", "Movie Offers"],
        "categories": ["shopping", "entertainment"],
        "image_url": "https://example.com/sbi-simplyclick.jpg",
        "apply_link": "https://example.com/apply/sbi-simplyclick",
    },
    {
        "id": 3,
        "name": "ICICI Amazon Pay",
        "issuer": "ICICI Bank",
        "joining_fee": 0,
        "annual_fee": 0,
        "reward_type": "Cashback",
        "reward_rate": "5% on Amazon, 2% on fuel",
        "min_income": 25000,
        "min_credit_score": 650,
        "perks": ["No Annual Fee", "Amazon Prime Benefits", "Fuel Surcharge Waiver"],
        "categories": ["shopping", "fuel"],
        "image_url": "https://example.com/icici-amazon.jpg",
        "apply_link": "https://example.com/apply/icici-amazon",
    },
    {
        "id": 4,
        "name": "Axis Magnus",
        "issuer": "Axis Bank",
        "joining_fee": 10000,
        "annual_fee": 10000,
        "reward_type": "Travel Points",
        "reward_rate": "12 Edge Miles per Rs.200",
        "min_income": 250000,
        "min_credit_score": 750,
        "perks": [
            "Priority Pass",
            "Golf Access",
            "Luxury Hotel Benefits",
            "Concierge Service",
        ],
        "categories": ["travel", "luxury", "dining"],
        "image_url": "https://example.com/axis-magnus.jpg",
        "apply_link": "https://example.com/apply/axis-magnus",
    },
    {
        "id": 5,
        "name": "IndianOil HDFC",
        "issuer": "HDFC Bank",
        "joining_fee": 500,
        "annual_fee": 500,
        "reward_type": "Fuel Points",
        "reward_rate": "5% cashback on fuel",
        "min_income": 30000,
        "min_credit_score": 700,
        "perks": ["Fuel Surcharge Waiver", "Zero Annual Fee on spending"],
        "categories": ["fuel"],
        "image_url": "https://example.com/indianoil-hdfc.jpg",
        "apply_link": "https://example.com/apply/indianoil-hdfc",
    },
    {
        "id": 6,
        "name": "American Express Gold",
        "issuer": "American Express",
        "joining_fee": 4500,
        "annual_fee": 4500,
        "reward_type": "Membership Rewards",
        "reward_rate": "4x points on dining, 2x on other spends",
        "min_income": 180000,
        "min_credit_score": 750,
        "perks": [
            "Dining Privileges",
            "Taj Experiences",
            "Travel Insurance",
            "Global Lounge Access",
        ],
        "categories": ["dining", "travel", "luxury"],
        "image_url": "https://example.com/amex-gold.jpg",
        "apply_link": "https://example.com/apply/amex-gold",
    },
    {
        "id": 7,
        "name": "Flipkart Axis Bank",
        "issuer": "Axis Bank",
        "joining_fee": 500,
        "annual_fee": 500,
        "reward_type": "Cashback",
        "reward_rate": "5% on Flipkart, 4% on travel",
        "min_income": 35000,
        "min_credit_score": 700,
        "perks": [
            "Flipkart Plus",
            "Zero Annual Fee on spending Rs.2L",
            "Travel Benefits",
        ],
        "categories": ["shopping", "travel"],
        "image_url": "https://example.com/flipkart-axis.jpg",
        "apply_link": "https://example.com/apply/flipkart-axis",
    },
    {
        "id": 8,
        "name": "Standard Chartered Ultimate",
        "issuer": "Standard Chartered",
        "joining_fee": 5000,
        "annual_fee": 5000,
        "reward_type": "Cashback",
        "reward_rate": "3.3% unlimited cashback",
        "min_income": 200000,
        "min_credit_score": 750,
        "perks": [
            "Unlimited Cashback",
            "Airport Lounge",
            "Golf Access",
            "No Categories",
        ],
        "categories": ["general", "travel", "shopping"],
        "image_url": "https://example.com/sc-ultimate.jpg",
        "apply_link": "https://example.com/apply/sc-ultimate",
    },
    {
        "id": 9,
        "name": "Citi Cashback",
        "issuer": "Citi Bank",
        "joining_fee": 500,
        "annual_fee": 500,
        "reward_type": "Cashback",
        "reward_rate": "5% on groceries, utilities, telecom",
        "min_income": 30000,
        "min_credit_score": 700,
        "perks": ["Grocery Rewards", "Bill Payment Benefits", "Utility Cashback"],
        "categories": ["groceries", "utilities"],
        "image_url": "https://example.com/citi-cashback.jpg",
        "apply_link": "https://example.com/apply/citi-cashback",
    },
    {
        "id": 10,
        "name": "HSBC Visa Platinum",
        "issuer": "HSBC",
        "joining_fee": 1500,
        "annual_fee": 1500,
        "reward_type": "Reward Points",
        "reward_rate": "2 points per Rs.150",
        "min_income": 100000,
        "min_credit_score": 720,
        "perks": [
            "Lounge Access",
            "Travel Benefits",
            "Dining Offers",
            "Movie Discounts",
        ],
        "categories": ["travel", "dining", "entertainment"],
        "image_url": "https://example.com/hsbc-platinum.jpg",
        "apply_link": "https://example.com/apply/hsbc-platinum",
    },
    {
        "id": 11,
        "name": "HDFC Millennia",
        "issuer": "HDFC Bank",
        "joining_fee": 1000,
        "annual_fee": 1000,
        "reward_type": "Cashback",
        "reward_rate": "5% on shopping, 2.5% on others",
        "min_income": 35000,
        "min_credit_score": 700,
        "perks": ["Online Shopping Rewards", "PayZapp Benefits", "SmartBuy Rewards"],
        "categories": ["shopping", "general"],
        "image_url": "https://example.com/hdfc-millennia.jpg",
        "apply_link": "https://example.com/apply/hdfc-millennia",
    },
    {
        "id": 12,
        "name": "YES First Exclusive",
        "issuer": "YES Bank",
        "joining_fee": 2499,
        "annual_fee": 2499,
        "reward_type": "Reward Points",
        "reward_rate": "6 points per Rs.200",
        "min_income": 120000,
        "min_credit_score": 720,
        "perks": ["Airport Lounge", "Dining Privileges", "Movie Benefits", "Golf"],
        "categories": ["travel", "dining", "entertainment"],
        "image_url": "https://example.com/yes-first.jpg",
        "apply_link": "https://example.com/apply/yes-first",
    },
    {
        "id": 13,
        "name": "Kotak Royale Signature",
        "issuer": "Kotak Mahindra Bank",
        "joining_fee": 999,
        "annual_fee": 999,
        "reward_type": "Reward Points",
        "reward_rate": "4 points per Rs.150",
        "min_income": 100000,
        "min_credit_score": 720,
        "perks": ["Lounge Access", "Golf Program", "Travel Benefits"],
        "categories": ["travel", "luxury"],
        "image_url": "https://example.com/kotak-royale.jpg",
        "apply_link": "https://example.com/apply/kotak-royale",
    },
    {
        "id": 14,
        "name": "RBL ShopRite",
        "issuer": "RBL Bank",
        "joining_fee": 0,
        "annual_fee": 0,
        "reward_type": "Cashback",
        "reward_rate": "5% on groceries and supermarkets",
        "min_income": 25000,
        "min_credit_score": 650,
        "perks": ["No Annual Fee", "Grocery Rewards", "Supermarket Cashback"],
        "categories": ["groceries"],
        "image_url": "https://example.com/rbl-shoprite.jpg",
        "apply_link": "https://example.com/apply/rbl-shoprite",
    },
    {
        "id": 15,
        "name": "IndusInd Legend",
        "issuer": "IndusInd Bank",
        "joining_fee": 10000,
        "annual_fee": 10000,
        "reward_type": "Travel Points",
        "reward_rate": "1.5 reward points per Rs.100",
        "min_income": 250000,
        "min_credit_score": 750,
        "perks": ["Priority Pass", "Golf Privileges", "Concierge", "Luxury Benefits"],
        "categories": ["travel", "luxury", "dining"],
        "image_url": "https://example.com/indusind-legend.jpg",
        "apply_link": "https://example.com/apply/indusind-legend",
    },
    {
        "id": 16,
        "name": "BOB Premier",
        "issuer": "Bank of Baroda",
        "joining_fee": 499,
        "annual_fee": 499,
        "reward_type": "Reward Points",
        "reward_rate": "4 points per Rs.100",
        "min_income": 30000,
        "min_credit_score": 700,
        "perks": ["Fuel Surcharge Waiver", "Dining Benefits", "Travel Discounts"],
        "categories": ["fuel", "dining", "travel"],
        "image_url": "https://example.com/bob-premier.jpg",
        "apply_link": "https://example.com/apply/bob-premier",
    },
    {
        "id": 17,
        "name": "AU Bank LIT",
        "issuer": "AU Small Finance Bank",
        "joining_fee": 0,
        "annual_fee": 0,
        "reward_type": "Cashback",
        "reward_rate": "5% on utilities, 1% on others",
        "min_income": 20000,
        "min_credit_score": 650,
        "perks": ["No Annual Fee", "Utility Cashback", "Mobile Recharge Benefits"],
        "categories": ["utilities", "general"],
        "image_url": "https://example.com/au-lit.jpg",
        "apply_link": "https://example.com/apply/au-lit",
    },
    {
        "id": 18,
        "name": "IDFC FIRST WOW",
        "issuer": "IDFC FIRST Bank",
        "joining_fee": 0,
        "annual_fee": 0,
        "reward_type": "Reward Points",
        "reward_rate": "10x on dining, 5x on travel",
        "min_income": 25000,
        "min_credit_score": 650,
        "perks": ["No Annual Fee", "Dining Rewards", "Travel Benefits"],
        "categories": ["dining", "travel"],
        "image_url": "https://example.com/idfc-wow.jpg",
        "apply_link": "https://example.com/apply/idfc-wow",
    },
    {
        "id": 19,
        "name": "Federal Celesta",
        "issuer": "Federal Bank",
        "joining_fee": 499,
        "annual_fee": 499,
        "reward_type": "Cashback",
        "reward_rate": "10% on dining, 5% on fuel",
        "min_income": 30000,
        "min_credit_score": 700,
        "perks": ["Dining Cashback", "Fuel Benefits", "Movie Discounts"],
        "categories": ["dining", "fuel", "entertainment"],
        "image_url": "https://example.com/federal-celesta.jpg",
        "apply_link": "https://example.com/apply/federal-celesta",
    },
    {
        "id": 20,
        "name": "PNB Rupay Select",
        "issuer": "Punjab National Bank",
        "joining_fee": 0,
        "annual_fee": 0,
        "reward_type": "Reward Points",
        "reward_rate": "2 points per Rs.100",
        "min_income": 20000,
        "min_credit_score": 650,
        "perks": ["No Annual Fee", "Lounge Access", "Fuel Surcharge Waiver"],
        "categories": ["general", "travel"],
        "image_url": "https://example.com/pnb-rupay.jpg",
        "apply_link": "https://example.com/apply/pnb-rupay",
    },
]


class ConversationAgent:
    def __init__(self):
        self.states = {
            "greeting": "What is your approximate monthly income in rupees? (e.g., 50000)",
            "fuel": "How much do you spend on fuel monthly? (Type: high, medium, low, or none)",
            "travel": "What about travel expenses? (Type: high, medium, low, or none)",
            "groceries": "How much do you spend on groceries? (Type: high, medium, low, or none)",
            "dining": "What about dining out? (Type: high, medium, low, or none)",
            "shopping": "How much do you shop online? (Type: high, medium, low, or none)",
            "benefits": "What benefits matter most to you? (Choose: cashback, travel, lounge, rewards)",
            "creditScore": "What's your approximate credit score? (Type a number like 750, or 'unknown')",
        }

    def get_next_state(self, current_state: str) -> Optional[str]:
        state_order = [
            "greeting",
            "fuel",
            "travel",
            "groceries",
            "dining",
            "shopping",
            "benefits",
            "creditScore",
        ]
        try:
            current_index = state_order.index(current_state)
            if current_index < len(state_order) - 1:
                return state_order[current_index + 1]
        except ValueError:
            pass
        return None

    def validate_input(self, message: str, state: str) -> bool:
        message_lower = message.lower()

        if state == "greeting":
            try:
                income = int("".join(filter(str.isdigit, message)))
                return income > 10000
            except:
                return False

        elif state in ["fuel", "travel", "groceries", "dining", "shopping"]:
            return message_lower in ["high", "medium", "low", "none"]

        elif state == "benefits":
            valid_benefits = ["cashback", "travel", "lounge", "rewards"]
            words = message_lower.split()
            return any(benefit in words for benefit in valid_benefits)

        elif state == "creditScore":
            if message_lower == "unknown":
                return True
            try:
                score = int("".join(filter(str.isdigit, message)))
                return 300 <= score <= 900
            except:
                return False

        return False

    def process_message(self, message: str, state: str, user_data: UserProfile) -> Dict:
        message_lower = message.lower()

        if not self.validate_input(message, state):
            return {
                "response": f"Invalid input. {self.states[state]}",
                "next_state": state,
                "updated_data": user_data.dict(),
                "is_complete": False,
            }

   
        updated_data = user_data.dict()

        if state == "greeting":
            income = int("".join(filter(str.isdigit, message)))
            updated_data["income"] = income
            confirmation = f"Great! Monthly income noted as ₹{income:,}."

        elif state in ["fuel", "travel", "groceries", "dining", "shopping"]:
            updated_data[state] = message_lower
            confirmation = "Got it!"

        elif state == "benefits":
            valid_benefits = ["cashback", "travel", "lounge", "rewards"]
            benefits = [b for b in valid_benefits if b in message_lower.split()]
            updated_data["benefits"] = benefits
            confirmation = "Great choices!"

        elif state == "creditScore":
            if message_lower == "unknown":
                score = 700
            else:
                score = int("".join(filter(str.isdigit, message)))
            updated_data["credit_score"] = score
            confirmation = "Perfect! Let me analyze your profile..."

        next_state = self.get_next_state(state)
        is_complete = next_state is None

        response_parts = [confirmation]
        if not is_complete:
            response_parts.append(self.states[next_state])

        return {
            "response": " ".join(response_parts),
            "next_state": next_state if not is_complete else "complete",
            "updated_data": updated_data,
            "is_complete": is_complete,
        }



class RecommendationEngine:
    @staticmethod
    def calculate_score(card: Dict, user_data: UserProfile) -> tuple:
        score = 0
        reasons = []

  
        annual_income = (user_data.income or 0) * 12
        if annual_income >= card["min_income"]:
            score += 30
            reasons.append(f"Matches your income requirement (₹{annual_income:,}/year)")


        credit_score = user_data.credit_score or 700
        if credit_score >= card["min_credit_score"]:
            score += 20
            reasons.append(f"Your credit score ({credit_score}) qualifies")


        spending_map = {
            "fuel": user_data.fuel,
            "travel": user_data.travel,
            "groceries": user_data.groceries,
            "dining": user_data.dining,
            "shopping": user_data.shopping,
        }

        for category in card["categories"]:
            if category in spending_map and spending_map[category] in [
                "high",
                "medium",
            ]:
                score += 15
                reasons.append(
                    f"Great for {category} (your {spending_map[category]} spending)"
                )


        if user_data.benefits:
            for benefit in user_data.benefits:
                if benefit == "cashback" and card["reward_type"] == "Cashback":
                    score += 20
                    reasons.append("Offers your preferred cashback rewards")
                    break
                elif benefit == "travel" and card["reward_type"] in [
                    "Travel Points",
                    "Membership Rewards",
                ]:
                    score += 20
                    reasons.append("Perfect for travel rewards")
                    break
                elif benefit == "lounge" and any(
                    "Lounge" in perk for perk in card["perks"]
                ):
                    score += 15
                    reasons.append("Includes airport lounge access")
                    break


        if card["joining_fee"] == 0:
            score += 10
            reasons.append("No joining fee")
        elif card["joining_fee"] < 1000:
            score += 5

        estimated_reward = RecommendationEngine.calculate_reward(card, user_data)
        reward_score = min(estimated_reward / 1000, 20)
        score += reward_score

        if estimated_reward > 5000:
            reasons.append(f"High reward potential: ₹{int(estimated_reward):,}/year")

        return score, reasons, estimated_reward

    @staticmethod
    def calculate_reward(card: Dict, user_data: UserProfile) -> float:
        monthly_income = user_data.income or 50000
        estimated_monthly_spend = monthly_income * 0.4
        annual_spend = estimated_monthly_spend * 12

        if card["reward_type"] == "Cashback":
            if "5%" in card["reward_rate"]:
                return annual_spend * 0.05
            elif "3.3%" in card["reward_rate"]:
                return annual_spend * 0.033
            elif "2%" in card["reward_rate"]:
                return annual_spend * 0.02
            else:
                return annual_spend * 0.025
        elif card["reward_type"] in ["Travel Points", "Membership Rewards"]:
            return annual_spend * 0.04
        else:
            return annual_spend * 0.025

    @staticmethod
    def get_recommendations(user_data: UserProfile) -> List[Dict]:
        scored_cards = []

        for card in CARD_DATABASE:
            score, reasons, estimated_reward = RecommendationEngine.calculate_score(
                card, user_data
            )

            scored_cards.append(
                {
                    **card,
                    "score": score,
                    "reasons": reasons,
                    "estimated_reward": int(estimated_reward),
                }
            )


        scored_cards.sort(key=lambda x: x["score"], reverse=True)
        return scored_cards[:5]


# API Endpoints
agent = ConversationAgent()


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML (rule-based advisor)"""
    with open("index.html", "r") as f:
        return f.read()


@app.get("/llm", response_class=HTMLResponse)
async def llm_root():
    """Serve the LLM-powered interface HTML"""
    with open("llm_index.html", "r",encoding="utf-8") as f:
        return f.read()


@app.post("/chat")
async def chat(chat_msg: ChatMessage):
    """Process chat message and return response"""
    try:
        result = agent.process_message(
            chat_msg.message, chat_msg.conversation_state, chat_msg.user_data
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/recommendations")
async def get_recommendations(user_data: UserProfile):
    """Get card recommendations based on user profile"""
    try:
        recommendations = RecommendationEngine.get_recommendations(user_data)
        return {"recommendations": recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cards")
async def get_all_cards():
    """Get all available credit cards"""
    return {"cards": CARD_DATABASE}


@app.get("/cards/{card_id}")
async def get_card(card_id: int):
    """Get specific card details"""
    card = next((c for c in CARD_DATABASE if c["id"] == card_id), None)
    if not card:
        raise HTTPException(status_code=404, detail="Card not found")
    return card



class LLMChatMessage(BaseModel):
    message: str
    user_profile: Optional[Dict] = None
    use_llm: bool = True


@app.post("/llm-advisor")
async def llm_advisor(chat_msg: LLMChatMessage):
    """
    Get AI-powered advice using LangChain and Google Gemini.
    This endpoint uses LLM to provide intelligent, context-aware responses.
    """
    try:
        from llm_integration import get_advisor

        advisor = get_advisor()
        response = advisor.get_advice(chat_msg.message, chat_msg.user_profile)

        return {
            "response": response,
            "powered_by": "LangChain + Google Gemini",
            "timestamp": datetime.now().isoformat(),
        }
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="LLM integration not properly installed. Please install langchain and google-generativeai.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/llm-recommendations")
async def llm_recommendations(user_data: UserProfile):
    """
    Get LLM-enhanced recommendations with latest market insights.
    Combines algorithmic scoring with AI analysis and web data.
    """
    try:
        from llm_integration import get_advisor

        advisor = get_advisor()
        recommendations = advisor.get_recommendations_llm(user_data.dict())

        return recommendations
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="LLM integration not properly installed. Please install langchain and google-generativeai.",
        )
    except Exception as e:
        logger.exception(f"Error in LLM recommendations {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/llm-status")
async def llm_status():
    """Check if LLM integration is properly configured."""
    try:
        import os

        api_key = os.getenv("GOOGLE_API_KEY")

        return {
            "status": "configured" if api_key else "not_configured",
            "model": "gemini-2.0-flash",
            "framework": "LangChain",
            "message": "Set GOOGLE_API_KEY environment variable to enable LLM features",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
