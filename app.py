import os

from flask import Flask, render_template
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Create the Flask application
app = Flask(__name__, instance_relative_config=True)

# Ensure the instance directory exists
os.makedirs(app.instance_path, exist_ok=True)

# Load the secret key
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY")

if not app.config["SECRET_KEY"]:
    raise RuntimeError(
        "SECRET_KEY is missing. Add it to your .env file."
    )


@app.route("/")
def home():
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True)
