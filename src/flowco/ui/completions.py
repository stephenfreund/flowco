# import threading
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import openai
# import os

# from flowco.util.output import log

# # Initialize Flask app
# app = Flask(__name__)

# # Enable CORS for all routes and origins
# CORS(app)

# # Set your OpenAI API key
# openai.api_key = os.getenv('OPENAI_API_KEY')

# # Define the route for GET requests
# @app.route('/complete', methods=['GET'])
# def complete_text():
#     """
#     Endpoint to handle text completion requests.
#     Expects a 'text' parameter in the query string.
#     """
#     # Retrieve 'text' parameter from query string
#     user_text = request.args.get('text')

#     if not user_text:
#         return jsonify({"error": "No 'text' parameter provided."}), 400

#     try:
#         completion = openai.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": "You are a helpful assistant."},
#                 {
#                     "role": "user",
#                     "content": user_text
#                 }
#             ],
#             max_tokens=150
#         )

#         # Extract the generated text
#         generated_text = completion.choices[0].message.content

#         # Return the generated text as JSON
#         return jsonify({"response": generated_text}), 200

#     except Exception as e:
#         # Handle exceptions, such as API errors
#         return jsonify({"error": str(e)}), 500

# def run_server(host='0.0.0.0', port=8421):
#     """
#     Function to run the Flask server.
#     """
#     app.run(host=host, port=port, debug=True, use_reloader=False)

# def start_server_in_thread(host='0.0.0.0', port=8421):
#     """
#     Starts the Flask server in a separate daemon thread.
#     """
#     server_thread = threading.Thread(target=run_server, args=(host, port))
#     server_thread.daemon = True  # Ensures the thread exits when main program does
#     server_thread.start()
#     log(f"Server started on http://{host}:{port}")

# if __name__ == "__main__":
#     # Start the server in a separate thread
#     start_server_in_thread(host='0.0.0.0', port=8421)

#     # Keep the main thread alive to keep the server running
#     try:
#         while True:
#             pass
#     except KeyboardInterrupt:
#         print("\nServer shutting down.")
