# YouTube Watch Time Calculator

A Python script to calculate and visualize your total YouTube watch time based on your watch history.

## Features

- Processes YouTube watch history from Google Takeout JSON files
- Fetches video details (duration, view count, etc.) from the YouTube API
- Stores data in a SQLite database for persistence and resumable processing
- Calculates comprehensive statistics about your viewing habits
- Generates visualizations of your watch patterns
- Exports data to CSV files for further analysis

## Requirements

- Python 3.6+
- YouTube API Key (stored in `.env` file)
- Google Takeout watch history data

## Installation

1. Clone this repository
2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```
4. Create a `.env` file with your YouTube API key:
   ```
   YOUTUBE_API_KEY=your_api_key_here
   ```

## Usage

### Basic Usage

```bash
python youtube_watchtime.py --history path/to/watch-history.json
```

### Options

- `--history`: Path to the YouTube watch history JSON file (default: watch-history.json)
- `--db`: Path to the SQLite database file (default: youtube_watchtime.db)
- `--export`: Export results to CSV files
- `--stats-only`: Display statistics without processing new data
- `--batch-size`: Batch size for API requests (max 50)
- `--plot`: Generate visualizations of the statistics

### Examples

Process watch history and display statistics:

```bash
python youtube_watchtime.py --history watch-history.json
```

Process watch history, display statistics, and generate plots:

```bash
python youtube_watchtime.py --history watch-history.json --plot
```

Display statistics without processing new data:

```bash
python youtube_watchtime.py --stats-only
```

Export data to CSV files:

```bash
python youtube_watchtime.py --export
```

## Getting Your Watch History

1. Go to [Google Takeout](https://takeout.google.com/)
2. Select only "YouTube and YouTube Music" and deselect all other services
3. Click "All YouTube data included" and deselect everything except "history"
4. Click "Next step" and create the export
5. Download the export when it's ready
6. Extract the ZIP file and find the `watch-history.json` file in the `Takeout/YouTube and YouTube Music/history/` directory

## Visualizations

The script can generate the following visualizations:

1. Video Length Distribution
2. Top Watched Channels
3. Viewing Pattern by Day of Week
4. Viewing Pattern by Hour of Day
5. Monthly Watch Statistics
6. Video Popularity Distribution
7. Summary Dashboard

To generate visualizations, use the `--plot` option:

```bash
python youtube_watchtime.py --plot
```

The plots will be saved to the `youtube_stats/plots` directory.

## Statistics

The script calculates and displays the following statistics:

- Total videos watched
- Unique videos watched
- Unique channels watched
- Total watch time
- Average video length
- Video length distribution
- Top 10 most watched channels
- Viewing pattern by day of week
- Viewing pattern by hour of day
- Monthly watch statistics
- Video popularity distribution by view count

## How It Works

1. The script loads your watch history from the JSON file
2. It extracts video IDs and channel information from the watch history
3. It uses the YouTube API to fetch video details (duration, view count, etc.)
4. It stores all data in a SQLite database for persistence
5. It calculates statistics based on the processed data
6. It displays the statistics and optionally generates visualizations

The script processes your watch history in batches to respect YouTube API rate limits. It also stores progress in the database so processing can be resumed if interrupted.

## Notes

- The YouTube API has a daily quota limit. If you have a large watch history, you may need to run the script over multiple days.
- The script only processes videos that are still available on YouTube. Deleted or private videos will be skipped.
- The accuracy of the statistics depends on the completeness of your watch history.
