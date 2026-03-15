"""
Dev server with live reload: browser refreshes when you save templates or this file.
Run: python run_live.py
Open: http://localhost:5500
"""
from flask import Flask, Response, render_template, jsonify
from livereload import Server

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.debug = True


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
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
    server = Server(app.wsgi_app)
    server.watch("templates/")
    server.watch("run_live.py")
    server.serve(port=5500, host="127.0.0.1")
