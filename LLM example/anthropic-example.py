import anthropic
import os

# No need for dotenv - using built-in os.getenv()

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found in environment variables")

client = anthropic.Anthropic(api_key=api_key)

# Define the categorization tool
tools = [
    {
        "name": "categorize_interruption",
        "description": "Categorize a machine interruption statement from a galvanizing line",
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["laitevika", "laatuongelma", "materiaaliongelma", "suunniteltu_huolto", "kayttajavirhe"],
                    "description": "The category of the interruption"
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "description": "Confidence score for the categorization"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Brief explanation for the categorization in Finnish"
                }
            },
            "required": ["category", "confidence", "reasoning"]
        }
    }
]

def categorize_statement(statement):
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        tools=tools,
        tool_choice={"type": "tool", "name": "categorize_interruption"},
        messages=[
            {
                "role": "user",
                "content": f"""Analysoi seuraava galvanointilinjan keskeytystilanne ja luokittele se yhteen seuraavista kategorioista:

- laitevika: Laitteiden rikkoutumiset, tekniset viat
- laatuongelma: Pinnoitteen laatu, paksuus, tarttuvuus
- materiaaliongelma: Kemikaalit, sinkkitaso, kappaleiden kunto  
- suunniteltu_huolto: Määräaikaishuollot, puhdistukset
- kayttajavirhe: Inhimilliset virheet, väärät asetukset

Tilanne: "{statement}"

Käytä categorize_interruption-työkalua vastauksessasi."""
            }
        ]
    )
    
    return message.content[0].input

# Test it
statement = "Upotuspumppu numero 3 pysähtynyt, öljynpaine liian matala"
result = categorize_statement(statement)
print(result)