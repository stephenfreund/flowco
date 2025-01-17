import boto3
import time
import datetime
import argparse
from botocore.exceptions import ClientError


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Fetch and tail AWS CloudWatch Logs for a specified log group."
    )
    parser.add_argument(
        "--log-group",
        default="/ecs/flowco-task",
        help="Name of the AWS CloudWatch log group (e.g., /ecs/flowco-task).",
    )
    parser.add_argument(
        "--stream-prefix",
        default=None,
        help="(Optional) Prefix of the log streams to filter by.",
    )
    parser.add_argument(
        "--initial-events",
        type=int,
        default=500,
        help="Number of initial log events to fetch (default: 500).",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=5,
        help="Seconds between polling for new logs (default: 5).",
    )
    return parser.parse_args()


def get_latest_log_stream(client, log_group_name, stream_prefix=None):
    try:
        kwargs = {
            "logGroupName": log_group_name,
            "orderBy": "LastEventTime",
            "descending": True,
            "limit": 1,
        }
        if stream_prefix:
            kwargs["logStreamNamePrefix"] = stream_prefix

        response = client.describe_log_streams(**kwargs)
        if response["logStreams"]:
            return response["logStreams"][0]["logStreamName"]
        else:
            print("No log streams found.")
            return None
    except ClientError as e:
        print(f"Error fetching log streams: {e}")
        return None


def fetch_initial_logs(client, log_group_name, log_stream_name, initial_events):
    try:
        response = client.get_log_events(
            logGroupName=log_group_name,
            logStreamName=log_stream_name,
            startFromHead=False,  # Get the latest events
            limit=initial_events,
        )
        events = response["events"]
        # Sort events by timestamp
        events.sort(key=lambda event: event["timestamp"])
        for event in events:
            timestamp = datetime.datetime.fromtimestamp(event["timestamp"] / 1000.0)
            print(f"{timestamp} - {event['message']}")
        return response["nextForwardToken"]
    except ClientError as e:
        print(f"Error fetching log events: {e}")
        return None


def stream_new_logs(client, log_group_name, log_stream_name, next_token, poll_interval):
    print("\n--- Starting live tail ---\n")
    while True:
        try:
            response = client.get_log_events(
                logGroupName=log_group_name,
                logStreamName=log_stream_name,
                startFromHead=False,
                nextToken=next_token,
                limit=100,
            )
            events = response["events"]
            if events:
                # Sort events by timestamp
                events.sort(key=lambda event: event["timestamp"])
                for event in events:
                    timestamp = datetime.datetime.fromtimestamp(
                        event["timestamp"] / 1000.0
                    )
                    print(f"{timestamp} - {event['message']}")
                next_token = response["nextForwardToken"]
            else:
                time.sleep(poll_interval)
        except ClientError as e:
            print(f"Error streaming log events: {e}")
            time.sleep(poll_interval)
        except KeyboardInterrupt:
            print("\nLive tail stopped by user.")
            break


def main():
    args = parse_arguments()

    client = boto3.client("logs")

    log_stream = get_latest_log_stream(
        client, log_group_name=args.log_group, stream_prefix=args.stream_prefix
    )
    if not log_stream:
        return

    next_token = fetch_initial_logs(
        client,
        log_group_name=args.log_group,
        log_stream_name=log_stream,
        initial_events=args.initial_events,
    )
    if not next_token:
        return

    stream_new_logs(
        client,
        log_group_name=args.log_group,
        log_stream_name=log_stream,
        next_token=next_token,
        poll_interval=args.poll_interval,
    )


if __name__ == "__main__":
    main()
