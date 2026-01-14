from supabase import create_client
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
load_dotenv(Path(__file__).parent / '.env.local', override=True)

supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

# Check how many games are in the database
response = supabase.table('games').select('game_id, game_date, status', count='exact').order('game_date', desc=True).limit(20).execute()

print(f"Total games in database: {response.count if response.count else 'Unknown'}")
print("\nMost recent games:")
for game in response.data[:10]:
    print(f"  {game['game_id']} - {game['game_date']} - {game.get('status', 'N/A')}")
