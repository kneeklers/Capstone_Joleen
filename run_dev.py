"""
Minimal dev server to preview the landing page layout (no camera or TensorFlow).
Run: python run_dev.py
Open: http://localhost:5000
"""
from flask import Flask, Response, render_template, jsonify

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    # Placeholder: 1x1 transparent pixel so the page layout works
    import base64
    pixel = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
    )
    return Response(pixel, mimetype="image/png")


@app.route("/api/logs")
def api_logs():
    return jsonify({"lines": []})


@app.route("/api/zones")
def api_zones():
    return jsonify({"zones": {}})


@app.route("/api/analysis")
def api_analysis():
    return jsonify({"analysis": False})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
