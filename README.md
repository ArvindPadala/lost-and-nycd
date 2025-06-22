# Lost & NYC’d – The Urban Lost-and-Found Oracle

An AI-powered real-time system to detect, track, and narrate **potentially lost items** in urban environments using webcams or live feeds. Inspired by NYC’s public space chaos, this project uses **Moondream’s vision model** to detect items like backpacks, phones, bottles, and wallets—and determines whether they've been unattended for too long.

---

## Features

-  Real-time object detection using Moondream Vision API
-  Object tracking across frames with bounding box persistence
-  Classifies items as **potentially lost** if unattended for ≥ 30 frames
-  Color-coded visual feedback (Yellow → Red → Blue)
-  Scene-level **poetic captions** (e.g., “A lone wallet resting by the café table...”)
-  CSV logging of lingering/lost objects
-  Saves the **final frame** with lost items highlighted
-  Auto-switches API keys on rate-limit (429) errors

---

##  How It Works

1. Initializes webcam and loads Moondream model
2. Detects key objects from list: `["backpack", "wallet", "phone", "bottle"]`
3. Tracks object bounding boxes across frames using IoU matching
4. If an object is present for more than 30 frames, it's marked 🔵 as **potentially lost**
5. Saves logs and final visual output

---

##  Installation

```bash
git clone https://github.com/your-username/lost-and-nycd.git
cd lost-and-nycd
pip install -r requirements.txt

##  File Structure

 lost-and-nycd/
├── detect_object.py            # Main script
├── lost_items_log.csv          # Output log file
├── final_lost_items_frame.jpg  # Output image on exit
├── final_lingering_summary.json # Object metadata
└── README.md 

## Example Output

[FRAME 47] Detected Now: 1 objects
Lingering items: ['bottle (ID: 6e34909e)', 'bottle (ID: ff7a2ed0)']
Potentially lost: bottle (ID: ff7a2ed0) for 30 frames
Scene Story: A blue water bottle stands alone near the subway steps, beside a crumpled napkin and shadows of passersby.

## Run the App

python detect_object.py


## Future Ideas

1. Integrate NYC 311 complaint data
2. Add location metadata (e.g., subway, park, etc.)
3. Deploy to Raspberry Pi or Jetson Nano for edge AI

 ## License

Free to use, modify, and contribute.
Let me know if you’d like:
Happy publishing!

