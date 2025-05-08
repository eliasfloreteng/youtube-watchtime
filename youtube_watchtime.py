#!/usr/bin/env python3
"""
YouTube Watch Time Calculator

This script calculates the total watch time and provides statistics based on YouTube watch history.
It supports resuming operations and stores data in a SQLite database to avoid recalculating.
"""

import json
import os
import re
import sqlite3
import time
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import isodate
import pandas as pd
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

# Regular expression to extract video ID from YouTube URL
YOUTUBE_URL_PATTERN = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/watch\?v=|youtu\.be\/)([a-zA-Z0-9_-]{11})"


class YouTubeWatchTimeCalculator:
    def __init__(self, history_file, db_file="youtube_watchtime.db", batch_size=50):
        """
        Initialize the calculator with the history file path and database name.

        Args:
            history_file (str): Path to the YouTube watch history JSON file
            db_file (str): Path to the SQLite database file for storing results
            batch_size (int): Number of videos to process in a single API call (max 50)
        """
        self.history_file = history_file
        self.db_file = db_file
        self.batch_size = min(batch_size, 50)  # YouTube API limit is 50
        self.youtube = build("youtube", "v3", developerKey=API_KEY)
        self.setup_database()

    def setup_database(self):
        """Set up the SQLite database and create necessary tables if they don't exist."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Create tables if they don't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                video_id TEXT PRIMARY KEY,
                title TEXT,
                duration INTEGER,  -- duration in seconds
                category_id TEXT,
                view_count INTEGER,
                like_count INTEGER,
                comment_count INTEGER,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channels (
                channel_id TEXT PRIMARY KEY,
                name TEXT,
                processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS watch_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id TEXT,
                channel_id TEXT,
                watched_at TIMESTAMP,
                FOREIGN KEY (video_id) REFERENCES videos(video_id),
                FOREIGN KEY (channel_id) REFERENCES channels(channel_id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS processing_status (
                id INTEGER PRIMARY KEY,
                last_processed_index INTEGER DEFAULT 0,
                total_entries INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        conn.close()

    def extract_video_id(self, url):
        """
        Extract the YouTube video ID from a URL.

        Args:
            url (str): YouTube video URL

        Returns:
            str: Video ID or None if not found
        """
        # Try regex pattern first
        match = re.search(YOUTUBE_URL_PATTERN, url)
        if match:
            return match.group(1)

        # As a fallback, try parsing the URL
        try:
            parsed_url = urlparse(url)
            if parsed_url.netloc in ("youtube.com", "www.youtube.com"):
                return parse_qs(parsed_url.query).get("v", [None])[0]
            elif parsed_url.netloc in ("youtu.be", "www.youtu.be"):
                return parsed_url.path.lstrip("/")
        except Exception:
            pass

        return None

    def load_watch_history(self):
        """
        Load the watch history from the JSON file.

        Returns:
            list: List of watch history entries
        """
        with open(self.history_file, "r", encoding="utf-8") as f:
            history = json.load(f)
        return history

    def get_unprocessed_videos(self, history):
        """
        Identify videos that haven't been processed yet.

        Args:
            history (list): Watch history data

        Returns:
            tuple: (unprocessed_entries, start_index, total_entries)
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        # Get the last processed index
        cursor.execute(
            "SELECT last_processed_index, total_entries FROM processing_status WHERE id = 1"
        )
        row = cursor.fetchone()

        start_index = 0
        if row:
            start_index = row[0]
            # If total entries has changed, we need to update
            if row[1] != len(history):
                cursor.execute(
                    "UPDATE processing_status SET total_entries = ? WHERE id = 1",
                    (len(history),),
                )
                conn.commit()
        else:
            # First run, initialize status
            cursor.execute(
                "INSERT INTO processing_status (id, last_processed_index, total_entries) VALUES (1, 0, ?)",
                (len(history),),
            )
            conn.commit()

        conn.close()

        # Return unprocessed entries
        return history[start_index:], start_index, len(history)

    def update_processing_status(self, processed_index):
        """
        Update the processing status in the database.

        Args:
            processed_index (int): The index up to which processing has been completed
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE processing_status 
            SET last_processed_index = ?, last_updated = CURRENT_TIMESTAMP 
            WHERE id = 1
        """,
            (processed_index,),
        )

        conn.commit()
        conn.close()

    def store_channel_info(self, channel_id, channel_name):
        """
        Store channel information in the database if it doesn't exist.

        Args:
            channel_id (str): YouTube channel ID
            channel_name (str): Channel name
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute("SELECT 1 FROM channels WHERE channel_id = ?", (channel_id,))
        if not cursor.fetchone():
            cursor.execute(
                """
                INSERT INTO channels (channel_id, name) VALUES (?, ?)
            """,
                (channel_id, channel_name),
            )

        conn.commit()
        conn.close()

    def store_watch_event(self, video_id, channel_id, watched_at):
        """
        Store a watch event in the database.

        Args:
            video_id (str): YouTube video ID
            channel_id (str): YouTube channel ID
            watched_at (str): ISO 8601 timestamp of when the video was watched
        """
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO watch_events (video_id, channel_id, watched_at)
            VALUES (?, ?, ?)
        """,
            (video_id, channel_id, watched_at),
        )

        conn.commit()
        conn.close()

    def fetch_video_details(self, video_ids):
        """
        Fetch video details from the YouTube API.

        Args:
            video_ids (list): List of YouTube video IDs

        Returns:
            dict: Dictionary mapping video IDs to their details
        """
        if not video_ids:
            return {}

        try:
            # Fetch video details
            response = (
                self.youtube.videos()
                .list(part="contentDetails,statistics,snippet", id=",".join(video_ids))
                .execute()
            )

            video_details = {}
            for item in response.get("items", []):
                video_id = item["id"]
                try:
                    duration_str = item["contentDetails"]["duration"]
                    duration_seconds = int(
                        isodate.parse_duration(duration_str).total_seconds()
                    )

                    # Get statistics (may not be available for all videos)
                    statistics = item.get("statistics", {})
                    view_count = int(statistics.get("viewCount", 0))
                    like_count = int(statistics.get("likeCount", 0))
                    comment_count = int(statistics.get("commentCount", 0))

                    # Get category ID
                    category_id = item["snippet"].get("categoryId", "")

                    video_details[video_id] = {
                        "title": item["snippet"]["title"],
                        "duration": duration_seconds,
                        "category_id": category_id,
                        "view_count": view_count,
                        "like_count": like_count,
                        "comment_count": comment_count,
                    }
                except (KeyError, ValueError) as e:
                    print(f"Error processing video {video_id}: {str(e)}")

            return video_details

        except HttpError as e:
            print(f"HTTP error occurred: {e}")
            time.sleep(10)  # Back off if rate limited
            return {}

    def store_video_details(self, video_details):
        """
        Store video details in the database.

        Args:
            video_details (dict): Dictionary mapping video IDs to their details
        """
        if not video_details:
            return

        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()

        for video_id, details in video_details.items():
            cursor.execute(
                """
                INSERT OR REPLACE INTO videos 
                (video_id, title, duration, category_id, view_count, like_count, comment_count) 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    video_id,
                    details["title"],
                    details["duration"],
                    details["category_id"],
                    details["view_count"],
                    details["like_count"],
                    details["comment_count"],
                ),
            )

        conn.commit()
        conn.close()

    def process_watch_history(self):
        """
        Process the watch history and store data in the database.
        This is the main function that handles all the processing.
        """
        print("Loading watch history...")
        history = self.load_watch_history()

        unprocessed_entries, start_index, total_entries = self.get_unprocessed_videos(
            history
        )

        if not unprocessed_entries:
            print("No new entries to process. All watch history has been processed.")
            return

        print(
            f"Processing {len(unprocessed_entries)} new entries (starting at index {start_index} of {total_entries})..."
        )

        # Process in batches to minimize API calls
        batch_video_ids = []
        current_index = start_index

        for i, entry in enumerate(
            tqdm(unprocessed_entries, desc="Processing watch history")
        ):
            try:
                # Extract information from the entry
                video_url = entry.get("titleUrl", "")
                video_id = self.extract_video_id(video_url)

                if not video_id:
                    current_index += 1
                    continue

                # Get channel info
                subtitles = entry.get("subtitles", [])
                channel_name = None
                channel_id = None
                if subtitles and len(subtitles) > 0:
                    channel_name = subtitles[0].get("name", "Unknown Channel")
                    channel_url = subtitles[0].get("url", "")
                    # Attempt to extract channel ID from URL
                    match = re.search(r"channel\/([^\/\?]+)", channel_url)
                    if match:
                        channel_id = match.group(1)
                    else:
                        channel_id = f"unknown_{hash(channel_name)}"

                # Store channel info
                if channel_id and channel_name:
                    self.store_channel_info(channel_id, channel_name)

                # Store watch event
                watched_at = entry.get("time", "")
                if video_id and channel_id and watched_at:
                    self.store_watch_event(video_id, channel_id, watched_at)

                # Add to batch for API processing
                batch_video_ids.append(video_id)

                # Process batch if it reaches the size limit
                if len(batch_video_ids) >= self.batch_size:
                    # Get unique video IDs (no duplicates)
                    unique_video_ids = list(set(batch_video_ids))
                    video_details = self.fetch_video_details(unique_video_ids)
                    self.store_video_details(video_details)
                    batch_video_ids = []

                    # Update status every batch
                    self.update_processing_status(current_index + 1)

                # Update current index
                current_index += 1

                # Small delay to avoid hitting API rate limits
                time.sleep(0.1)

            except Exception as e:
                print(f"Error processing entry {i + start_index}: {str(e)}")
                # Continue with next entry

        # Process any remaining videos
        if batch_video_ids:
            unique_video_ids = list(set(batch_video_ids))
            video_details = self.fetch_video_details(unique_video_ids)
            self.store_video_details(video_details)

        # Final update of processing status
        self.update_processing_status(current_index)

        print(f"Processed {current_index - start_index} entries successfully.")

    def calculate_statistics(self):
        """
        Calculate statistics from the processed data.

        Returns:
            dict: Dictionary containing various statistics
        """
        conn = sqlite3.connect(self.db_file)

        # Create a connection to the database for pandas
        df_videos = pd.read_sql_query("SELECT * FROM videos", conn)
        df_channels = pd.read_sql_query("SELECT * FROM channels", conn)
        df_watch_events = pd.read_sql_query("SELECT * FROM watch_events", conn)

        # Join to get complete information
        df_combined = pd.merge(df_watch_events, df_videos, on="video_id", how="left")
        df_combined = pd.merge(df_combined, df_channels, on="channel_id", how="left")

        # Convert watched_at to datetime with flexible format handling
        df_combined["watched_at"] = pd.to_datetime(
            df_combined["watched_at"], format="ISO8601"
        )

        # Initialize statistics dictionary
        stats = {}

        # Basic counts
        stats["total_videos"] = len(df_combined)
        stats["unique_videos"] = df_combined["video_id"].nunique()
        stats["unique_channels"] = df_combined["channel_id"].nunique()

        # Watch time statistics
        total_seconds = df_combined["duration"].sum()
        stats["total_watch_time_seconds"] = int(total_seconds)
        stats["total_watch_time_hours"] = round(
            total_seconds / 3600, 2
        )  # Convert to hours
        stats["total_watch_time_formatted"] = self.format_duration(total_seconds)

        if len(df_combined) > 0:
            stats["average_video_length_seconds"] = int(df_combined["duration"].mean())
            stats["average_video_length_formatted"] = self.format_duration(
                df_combined["duration"].mean()
            )

            # Video length distribution
            df_combined["length_category"] = pd.cut(
                df_combined["duration"],
                bins=[0, 300, 1200, 3600, float("inf")],
                labels=[
                    "Short (< 5 min)",
                    "Medium (5-20 min)",
                    "Long (20-60 min)",
                    "Very Long (> 60 min)",
                ],
            )
            stats["video_length_distribution"] = (
                df_combined["length_category"].value_counts().to_dict()
            )

        # Channel statistics
        if "name" in df_combined.columns:
            channel_stats = (
                df_combined.groupby("name")
                .agg({"video_id": "count", "duration": "sum"})
                .sort_values("video_id", ascending=False)
            )

            stats["top_channels"] = (
                channel_stats.head(10)
                .reset_index()
                .rename(
                    columns={"video_id": "watch_count", "duration": "total_seconds"}
                )
                .to_dict("records")
            )

            # Add formatted watch time
            for channel in stats["top_channels"]:
                channel["watch_time_formatted"] = self.format_duration(
                    channel["total_seconds"]
                )

        # Temporal statistics
        if "watched_at" in df_combined.columns:
            # Day of week
            df_combined["day_of_week"] = df_combined["watched_at"].dt.day_name()
            stats["day_of_week_distribution"] = (
                df_combined["day_of_week"].value_counts().to_dict()
            )

            # Hour of day
            df_combined["hour_of_day"] = df_combined["watched_at"].dt.hour
            stats["hour_of_day_distribution"] = (
                df_combined["hour_of_day"].value_counts().sort_index().to_dict()
            )

            # Month distribution
            df_combined["month_year"] = df_combined["watched_at"].dt.strftime("%Y-%m")
            monthly_stats = (
                df_combined.groupby("month_year")
                .agg({"video_id": "count", "duration": "sum"})
                .sort_index()
                .tail(12)
            )  # Last 12 months

            stats["monthly_stats"] = (
                monthly_stats.reset_index()
                .rename(
                    columns={"video_id": "watch_count", "duration": "total_seconds"}
                )
                .to_dict("records")
            )

            # Add formatted watch time
            for month in stats["monthly_stats"]:
                month["watch_time_formatted"] = self.format_duration(
                    month["total_seconds"]
                )

            # Quarter distribution
            df_combined["quarter"] = (
                df_combined["watched_at"].dt.to_period("Q").astype(str)
            )
            quarterly_stats = (
                df_combined.groupby("quarter")
                .agg({"video_id": "count", "duration": "sum"})
                .sort_index()
                # Include all quarters in the history
            )

            stats["quarterly_stats"] = (
                quarterly_stats.reset_index()
                .rename(
                    columns={"video_id": "watch_count", "duration": "total_seconds"}
                )
                .to_dict("records")
            )

            # Add formatted watch time
            for quarter in stats["quarterly_stats"]:
                quarter["watch_time_formatted"] = self.format_duration(
                    quarter["total_seconds"]
                )
                quarter["watch_time_hours"] = round(quarter["total_seconds"] / 3600, 2)

        # Video view statistics (if available)
        if "view_count" in df_combined.columns and df_combined["view_count"].sum() > 0:
            stats["avg_video_views"] = int(df_combined["view_count"].mean())
            stats["total_video_views"] = int(df_combined["view_count"].sum())

            # Popularity distribution
            df_combined["popularity_category"] = pd.cut(
                df_combined["view_count"],
                bins=[0, 1000, 10000, 100000, 1000000, float("inf")],
                labels=["< 1K", "1K-10K", "10K-100K", "100K-1M", "> 1M"],
            )
            stats["popularity_distribution"] = (
                df_combined["popularity_category"].value_counts().to_dict()
            )

        conn.close()
        return stats

    def format_duration(self, seconds):
        """
        Format duration in seconds to a human-readable string.

        Args:
            seconds (float): Duration in seconds

        Returns:
            str: Formatted duration string
        """
        seconds = int(seconds)
        days, seconds = divmod(seconds, 86400)
        hours, seconds = divmod(seconds, 3600)
        minutes, seconds = divmod(seconds, 60)

        parts = []
        if days > 0:
            parts.append(f"{days} day{'s' if days != 1 else ''}")
        if hours > 0:
            parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
        if minutes > 0:
            parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
        if seconds > 0 or not parts:
            parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

        return ", ".join(parts)

    def export_to_csv(self, output_dir="youtube_stats"):
        """
        Export the processed data to CSV files.

        Args:
            output_dir (str): Directory to save the CSV files
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        conn = sqlite3.connect(self.db_file)

        # Export videos data
        df_videos = pd.read_sql_query("SELECT * FROM videos", conn)
        df_videos.to_csv(os.path.join(output_dir, "videos.csv"), index=False)

        # Export channels data
        df_channels = pd.read_sql_query("SELECT * FROM channels", conn)
        df_channels.to_csv(os.path.join(output_dir, "channels.csv"), index=False)

        # Export watch events data
        df_watch_events = pd.read_sql_query("SELECT * FROM watch_events", conn)
        df_watch_events.to_csv(
            os.path.join(output_dir, "watch_events.csv"), index=False
        )

        # Create a combined dataset for easier analysis
        df_combined = pd.read_sql_query(
            """
            SELECT 
                w.id, w.video_id, w.channel_id, w.watched_at,
                v.title, v.duration, v.category_id, v.view_count, v.like_count, v.comment_count,
                c.name as channel_name
            FROM watch_events w
            LEFT JOIN videos v ON w.video_id = v.video_id
            LEFT JOIN channels c ON w.channel_id = c.channel_id
        """,
            conn,
        )

        df_combined.to_csv(os.path.join(output_dir, "combined_data.csv"), index=False)

        conn.close()
        print(f"Data exported to CSV files in '{output_dir}' directory.")

    def generate_plots(self, stats, output_dir="youtube_stats/plots"):
        """
        Generate visualizations of the statistics.

        Args:
            stats (dict): Statistics dictionary
            output_dir (str): Directory to save the plots
        """
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set the style for the plots
        sns.set(style="darkgrid")
        plt.rcParams.update({"font.size": 12})

        # 1. Video Length Distribution
        if "video_length_distribution" in stats:
            plt.figure(figsize=(12, 8))
            categories = list(stats["video_length_distribution"].keys())
            counts = list(stats["video_length_distribution"].values())

            # Sort by video length category
            order = [
                "Short (< 5 min)",
                "Medium (5-20 min)",
                "Long (20-60 min)",
                "Very Long (> 60 min)",
            ]
            sorted_data = [
                (cat, stats["video_length_distribution"].get(cat, 0))
                for cat in order
                if cat in stats["video_length_distribution"]
            ]
            categories, counts = zip(*sorted_data) if sorted_data else ([], [])

            ax = sns.barplot(x=list(categories), y=list(counts))
            plt.title("Video Length Distribution", fontsize=16)
            plt.xlabel("Video Length Category")
            plt.ylabel("Number of Videos")
            plt.xticks(rotation=45)

            # Add count labels on top of bars
            for i, count in enumerate(counts):
                ax.text(i, count + (max(counts) * 0.01), f"{count:,}", ha="center")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "video_length_distribution.png"))
            plt.close()
            print("Saved plot: video_length_distribution.png")

        # 2. Top Watched Channels
        if "top_channels" in stats and stats["top_channels"]:
            plt.figure(figsize=(14, 10))

            # Extract data for top 10 channels
            channels = [channel["name"] for channel in stats["top_channels"]]
            watch_counts = [channel["watch_count"] for channel in stats["top_channels"]]
            watch_times = [
                channel["total_seconds"] / 3600 for channel in stats["top_channels"]
            ]  # Convert to hours

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

            # Plot watch counts
            bars1 = sns.barplot(x=watch_counts, y=channels, ax=ax1, palette="viridis")
            ax1.set_title("Top 10 Channels by Watch Count", fontsize=16)
            ax1.set_xlabel("Number of Videos Watched")
            ax1.set_ylabel("Channel")

            # Add count labels
            for i, count in enumerate(watch_counts):
                ax1.text(
                    count + (max(watch_counts) * 0.01), i, f"{count:,}", va="center"
                )

            # Plot watch times
            bars2 = sns.barplot(x=watch_times, y=channels, ax=ax2, palette="viridis")
            ax2.set_title("Top 10 Channels by Watch Time (Hours)", fontsize=16)
            ax2.set_xlabel("Hours Watched")
            ax2.set_ylabel("Channel")

            # Add time labels
            for i, hours in enumerate(watch_times):
                ax2.text(
                    hours + (max(watch_times) * 0.01), i, f"{hours:.1f}", va="center"
                )

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "top_channels.png"))
            plt.close()
            print("Saved plot: top_channels.png")

        # 3. Viewing Pattern by Day of Week
        if "day_of_week_distribution" in stats:
            plt.figure(figsize=(12, 8))

            days_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            days_data = {
                day: stats["day_of_week_distribution"].get(day, 0) for day in days_order
            }

            ax = sns.barplot(
                x=list(days_data.keys()), y=list(days_data.values()), palette="viridis"
            )
            plt.title("Viewing Pattern by Day of Week", fontsize=16)
            plt.xlabel("Day of Week")
            plt.ylabel("Number of Videos Watched")

            # Add count labels
            for i, count in enumerate(days_data.values()):
                ax.text(
                    i,
                    count + (max(days_data.values()) * 0.01),
                    f"{count:,}",
                    ha="center",
                )

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "day_of_week_pattern.png"))
            plt.close()
            print("Saved plot: day_of_week_pattern.png")

        # 4. Viewing Pattern by Hour of Day
        if "hour_of_day_distribution" in stats:
            plt.figure(figsize=(14, 8))

            hours = list(range(24))
            counts = [stats["hour_of_day_distribution"].get(hour, 0) for hour in hours]

            ax = sns.barplot(x=hours, y=counts, palette="viridis")
            plt.title("Viewing Pattern by Hour of Day", fontsize=16)
            plt.xlabel("Hour of Day")
            plt.ylabel("Number of Videos Watched")
            plt.xticks(range(0, 24, 2), [f"{h:02d}:00" for h in range(0, 24, 2)])

            # Add count labels for hours with significant views
            for i, count in enumerate(counts):
                if count > max(counts) * 0.1:  # Only label bars with significant height
                    ax.text(
                        i,
                        count + (max(counts) * 0.01),
                        f"{count:,}",
                        ha="center",
                        fontsize=9,
                    )

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "hour_of_day_pattern.png"))
            plt.close()
            print("Saved plot: hour_of_day_pattern.png")

        # 5. Monthly Watch Statistics
        if "monthly_stats" in stats and stats["monthly_stats"]:
            plt.figure(figsize=(14, 8))

            months = [month["month_year"] for month in stats["monthly_stats"]]
            watch_counts = [month["watch_count"] for month in stats["monthly_stats"]]
            watch_times = [
                month["total_seconds"] / 3600 for month in stats["monthly_stats"]
            ]  # Convert to hours

            # Create a figure with two subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 16))

            # Plot watch counts
            bars1 = sns.barplot(x=months, y=watch_counts, ax=ax1, palette="viridis")
            ax1.set_title("Monthly Watch Count (Last 12 Months)", fontsize=16)
            ax1.set_xlabel("Month")
            ax1.set_ylabel("Number of Videos Watched")
            ax1.set_xticklabels(months, rotation=45)

            # Add count labels
            for i, count in enumerate(watch_counts):
                if count > max(watch_counts) * 0.05:  # Only label significant bars
                    ax1.text(
                        i,
                        count + (max(watch_counts) * 0.01),
                        f"{count:,}",
                        ha="center",
                        fontsize=9,
                    )

            # Plot watch times
            bars2 = sns.barplot(x=months, y=watch_times, ax=ax2, palette="viridis")
            ax2.set_title("Monthly Watch Time in Hours (Last 12 Months)", fontsize=16)
            ax2.set_xlabel("Month")
            ax2.set_ylabel("Hours Watched")
            ax2.set_xticklabels(months, rotation=45)

            # Add time labels
            for i, hours in enumerate(watch_times):
                if hours > max(watch_times) * 0.05:  # Only label significant bars
                    ax2.text(
                        i,
                        hours + (max(watch_times) * 0.01),
                        f"{hours:.1f}",
                        ha="center",
                        fontsize=9,
                    )

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "monthly_stats.png"))
            plt.close()
            print("Saved plot: monthly_stats.png")

        # 6. Video Popularity Distribution
        if "popularity_distribution" in stats:
            plt.figure(figsize=(12, 8))

            # Sort by view count category
            order = ["< 1K", "1K-10K", "10K-100K", "100K-1M", "> 1M"]
            sorted_data = [
                (cat, stats["popularity_distribution"].get(cat, 0))
                for cat in order
                if cat in stats["popularity_distribution"]
            ]
            categories, counts = zip(*sorted_data) if sorted_data else ([], [])

            ax = sns.barplot(x=list(categories), y=list(counts), palette="viridis")
            plt.title("Video Popularity Distribution (by View Count)", fontsize=16)
            plt.xlabel("View Count Category")
            plt.ylabel("Number of Videos")

            # Add count labels
            for i, count in enumerate(counts):
                ax.text(i, count + (max(counts) * 0.01), f"{count:,}", ha="center")

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "popularity_distribution.png"))
            plt.close()
            print("Saved plot: popularity_distribution.png")

        # 7. Quarterly Trend
        if "quarterly_stats" in stats and stats["quarterly_stats"]:
            plt.figure(figsize=(14, 10))

            # Extract data from quarterly stats
            quarters = [quarter["quarter"] for quarter in stats["quarterly_stats"]]
            watch_times = [
                quarter["watch_time_hours"] for quarter in stats["quarterly_stats"]
            ]

            # Sort quarters chronologically
            sorted_quarters = quarters
            sorted_hours = watch_times

            # Plot quarterly trend
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(x=sorted_quarters, y=sorted_hours, palette="viridis")
            plt.title("Quarterly Watch Time Trend", fontsize=16)
            plt.xlabel("Quarter")
            plt.ylabel("Hours Watched")
            plt.xticks(rotation=45)

            # Add hour labels
            for i, hours in enumerate(sorted_hours):
                ax.text(
                    i, hours + (max(sorted_hours) * 0.01), f"{hours:.1f}", ha="center"
                )

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, "quarterly_trend.png"))
            plt.close()
            print("Saved plot: quarterly_trend.png")

        # 8. Summary Dashboard
        plt.figure(figsize=(16, 12))

        # Create a 2x3 grid for the summary plots
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))

        # 1. Video Length Distribution (top left)
        if "video_length_distribution" in stats:
            order = [
                "Short (< 5 min)",
                "Medium (5-20 min)",
                "Long (20-60 min)",
                "Very Long (> 60 min)",
            ]
            sorted_data = [
                (cat, stats["video_length_distribution"].get(cat, 0))
                for cat in order
                if cat in stats["video_length_distribution"]
            ]
            categories, counts = zip(*sorted_data) if sorted_data else ([], [])

            sns.barplot(
                x=list(categories), y=list(counts), ax=axs[0, 0], palette="viridis"
            )
            axs[0, 0].set_title("Video Length Distribution")
            axs[0, 0].set_xlabel("Video Length")
            axs[0, 0].set_ylabel("Count")
            axs[0, 0].set_xticklabels(categories, rotation=45, fontsize=8)

        # 2. Top 5 Channels (top middle)
        if "top_channels" in stats and len(stats["top_channels"]) > 0:
            top_5_channels = stats["top_channels"][:5]
            channels = [channel["name"] for channel in top_5_channels]
            watch_counts = [channel["watch_count"] for channel in top_5_channels]

            sns.barplot(x=watch_counts, y=channels, ax=axs[0, 1], palette="viridis")
            axs[0, 1].set_title("Top 5 Channels")
            axs[0, 1].set_xlabel("Videos Watched")
            axs[0, 1].set_ylabel("Channel")

        # 3. Day of Week (top right)
        if "day_of_week_distribution" in stats:
            days_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            days_data = {
                day: stats["day_of_week_distribution"].get(day, 0) for day in days_order
            }

            sns.barplot(
                x=list(days_data.keys()),
                y=list(days_data.values()),
                ax=axs[0, 2],
                palette="viridis",
            )
            axs[0, 2].set_title("Viewing by Day of Week")
            axs[0, 2].set_xlabel("Day")
            axs[0, 2].set_ylabel("Count")
            axs[0, 2].set_xticklabels(days_data.keys(), rotation=45, fontsize=8)

        # 4. Hour of Day (bottom left)
        if "hour_of_day_distribution" in stats:
            hours = list(range(24))
            counts = [stats["hour_of_day_distribution"].get(hour, 0) for hour in hours]

            sns.barplot(x=hours, y=counts, ax=axs[1, 0], palette="viridis")
            axs[1, 0].set_title("Viewing by Hour of Day")
            axs[1, 0].set_xlabel("Hour")
            axs[1, 0].set_ylabel("Count")
            axs[1, 0].set_xticks(range(0, 24, 3))
            axs[1, 0].set_xticklabels(
                [f"{h:02d}:00" for h in range(0, 24, 3)], fontsize=8
            )

        # 5. Monthly Trend (bottom middle)
        if "monthly_stats" in stats and stats["monthly_stats"]:
            months = [month["month_year"] for month in stats["monthly_stats"]]
            watch_counts = [month["watch_count"] for month in stats["monthly_stats"]]

            sns.barplot(x=months, y=watch_counts, ax=axs[1, 1], palette="viridis")
            axs[1, 1].set_title("Monthly Trend")
            axs[1, 1].set_xlabel("Month")
            axs[1, 1].set_ylabel("Videos Watched")
            axs[1, 1].set_xticklabels(months, rotation=45, fontsize=8)

        # 6. Popularity Distribution (bottom right)
        if "popularity_distribution" in stats:
            order = ["< 1K", "1K-10K", "10K-100K", "100K-1M", "> 1M"]
            sorted_data = [
                (cat, stats["popularity_distribution"].get(cat, 0))
                for cat in order
                if cat in stats["popularity_distribution"]
            ]
            categories, counts = zip(*sorted_data) if sorted_data else ([], [])

            sns.barplot(
                x=list(categories), y=list(counts), ax=axs[1, 2], palette="viridis"
            )
            axs[1, 2].set_title("Video Popularity")
            axs[1, 2].set_xlabel("View Count")
            axs[1, 2].set_ylabel("Count")
            axs[1, 2].set_xticklabels(categories, rotation=45, fontsize=8)

        # Add a title to the entire figure
        fig.suptitle(
            f"YouTube Watch Statistics Dashboard\nTotal: {stats['total_videos']:,} videos, {stats['total_watch_time_formatted']}",
            fontsize=20,
        )

        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Adjust for the suptitle
        plt.savefig(os.path.join(output_dir, "dashboard.png"))
        plt.close()
        print("Saved plot: dashboard.png")

        print(f"\nAll plots have been saved to: {output_dir}")
        return output_dir

    def display_statistics(self, stats):
        """
        Display the calculated statistics in a user-friendly format.

        Args:
            stats (dict): Statistics dictionary
        """
        print("\n" + "=" * 80)
        print(" " * 30 + "YOUTUBE WATCH STATISTICS")
        print("=" * 80)

        # Basic counts
        print(f"\nTotal videos watched: {stats['total_videos']:,}")
        print(f"Unique videos watched: {stats['unique_videos']:,}")
        print(f"Unique channels watched: {stats['unique_channels']:,}")

        # Watch time
        total_hours = stats["total_watch_time_seconds"] / 3600
        print(
            f"\nTotal watch time: {stats['total_watch_time_formatted']} ({total_hours:.2f} hours)"
        )
        print(
            f"Average video length: {stats.get('average_video_length_formatted', 'N/A')}"
        )

        # Video length distribution
        if "video_length_distribution" in stats:
            print("\nVideo Length Distribution:")
            for category, count in stats["video_length_distribution"].items():
                print(f"  {category}: {count:,} videos")

        # Top channels
        if "top_channels" in stats:
            print("\nTop 10 Most Watched Channels:")
            for i, channel in enumerate(stats["top_channels"], 1):
                print(
                    f"  {i}. {channel['name']}: {channel['watch_count']:,} videos, {channel['watch_time_formatted']}"
                )

        # Viewing pattern by day of week
        if "day_of_week_distribution" in stats:
            print("\nViewing Pattern by Day of Week:")
            days_order = [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
            days_data = {
                day: stats["day_of_week_distribution"].get(day, 0) for day in days_order
            }
            for day, count in days_data.items():
                print(f"  {day}: {count:,} videos")

        # Viewing pattern by hour of day
        if "hour_of_day_distribution" in stats:
            print("\nViewing Pattern by Hour of Day:")
            for hour in range(24):
                count = stats["hour_of_day_distribution"].get(hour, 0)
                print(f"  {hour:02d}:00 - {hour:02d}:59: {count:,} videos")

        # Monthly statistics
        if "monthly_stats" in stats:
            print("\nMonthly Watch Statistics (Last 12 Months):")
            for month in stats["monthly_stats"]:
                print(
                    f"  {month['month_year']}: {month['watch_count']:,} videos, {month['watch_time_formatted']}"
                )

        # Video popularity
        if "popularity_distribution" in stats:
            print("\nVideo Popularity Distribution (by view count):")
            for category, count in stats["popularity_distribution"].items():
                print(f"  {category}: {count:,} videos")

        print("\n" + "=" * 80)


def main():
    """Main function to run the YouTube watch time calculator."""
    parser = argparse.ArgumentParser(
        description="Calculate YouTube watch time and statistics from watch history."
    )
    parser.add_argument(
        "--history",
        default="watch-history.json",
        help="Path to the YouTube watch history JSON file",
    )
    parser.add_argument(
        "--db", default="youtube_watchtime.db", help="Path to the SQLite database file"
    )
    parser.add_argument(
        "--export", action="store_true", help="Export results to CSV files"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Display statistics without processing new data",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for API requests (max 50)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualizations of the statistics",
    )

    args = parser.parse_args()

    calculator = YouTubeWatchTimeCalculator(
        history_file=args.history, db_file=args.db, batch_size=args.batch_size
    )

    if not args.stats_only:
        try:
            calculator.process_watch_history()
        except KeyboardInterrupt:
            print("\nProcessing interrupted by user. Progress has been saved.")

    stats = calculator.calculate_statistics()
    calculator.display_statistics(stats)

    if args.export:
        calculator.export_to_csv()

    if args.plot:
        plots_dir = calculator.generate_plots(stats)
        print(f"\nTo view the plots, open the files in the '{plots_dir}' directory.")


if __name__ == "__main__":
    main()
