# YouTube Watch Time Calculator

A Python script to calculate your total YouTube watch time and generate statistics based on your watch history. It fetches video durations from the YouTube API and provides detailed analytics about your viewing habits.

## Features

- Calculate total watch time from YouTube watch history
- Resume processing if interrupted (no recalculation needed)
- Generate comprehensive statistics:
  - Total and average watch time
  - Video length distribution
  - Most watched channels
  - Viewing patterns by day of week and hour of day
  - Monthly watch statistics
  - Video popularity distribution
- Export all data to CSV files for further analysis

## Requirements

- Python 3.6+
- Required Python packages (install via `pip install -r requirements.txt`):
  - google-api-python-client
  - python-dotenv
  - pandas
  - tqdm
  - isodate

## Setup

1. Clone this repository or download the script
2. Create a `.env` file with your YouTube API key:
   ```
   YOUTUBE_API_KEY=your_api_key_here
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python youtube_watchtime.py --history watch-history-sample.json
```

### Command Line Options

- `--history`: Path to the YouTube watch history JSON file (default: 'watch-history.json')
- `--db`: Path to the SQLite database file for storing processed data (default: 'youtube_watchtime.db')
- `--export`: Export results to CSV files in the 'youtube_stats' directory
- `--stats-only`: Display statistics without processing new watch history data
- `--batch-size`: Batch size for API requests (max 50, default: 50)

### Examples

Process the sample watch history:

```bash
python youtube_watchtime.py --history watch-history-sample.json
```

Process full watch history and export data:

```bash
python youtube_watchtime.py --history watch-history.json --export
```

Display statistics from previously processed data:

```bash
python youtube_watchtime.py --stats-only
```

## How to Get Your YouTube Watch History

1. Go to [Google Takeout](https://takeout.google.com/)
2. Select only "YouTube and YouTube Music" > "Deselect all" > select only "history"
3. Choose your preferred delivery method and export
4. Download and extract the archive
5. Find the `watch-history.json` file in the extracted data

## Notes

- Processing large watch histories may take time due to YouTube API rate limits
- The script saves progress to a SQLite database, so you can interrupt and resume processing at any time
- The sample file provided contains only a few entries for demonstration purposes
