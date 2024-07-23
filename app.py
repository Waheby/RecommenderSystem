from flask import Flask, request, jsonify
import pandas as pd
from flask_cors import CORS # CORS for handling Cross-Origin Resource Sharing
import pickle 
import model
from waitress import serve

# Create a Flask application instance
app = Flask(__name__)

# Enable CORS for all routes, allowing requests from any origin
CORS(app,resources={r"/*":{"origins":"*"}})

posts = pickle.load(open('post_list.pkl', 'rb'))
similarityPost = pickle.load(open('similarityPost_list.pkl', 'rb'))

users = pickle.load(open('user_list.pkl', 'rb'))
similarityUser = pickle.load(open('similarityUser_list.pkl', 'rb'))

# Define a route for handling HTTP GET requests to the root URL
@app.route('/', methods=['GET'])
def get_data():
    data = {
        "message":"API is Running"
    }
    return jsonify(data)

def recommendpost(post):
    index = posts[posts['_id'] == post].index[0]
    distances = sorted(list(enumerate(similarityPost[index])), reverse=True, key = lambda x: x[1])
    recommended_posts = []
    for i in distances[1:5]:
        recommended_posts.append(posts.iloc[i[0]]._id)
        print(posts.iloc[i[0]]._id)
    return recommended_posts

def recommenduser(user):
    index = users[users['username'] == user].index[0]
    distances = sorted(list(enumerate(similarityUser[index])), reverse=True, key = lambda x: x[1])
    recommended_users = []
    for i in distances[1:5]:
        recommended_users.append(users.iloc[i[0]].username)
        print(users.iloc[i[0]].username)
    return recommended_users
  
# Define a route for making predictions
@app.route('/recommend', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        print(data['content'])
        prediction = recommendpost(data['content'])
        return jsonify({'Recommendation': list(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})
    
# Define a route for making predictions
@app.route('/recommend-users', methods=['POST'])
def predictUsers():
    try:
        data = request.get_json()
        print("For User: " + data['content'])
        prediction = recommenduser(data['content'])
        return jsonify({'User Recommendation': list(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    serve(app, host="0.0.0.0", port=5000)