# Project Setup
- Clone the project
- Go to root directory using comman below
`cd rl_project`
- Create a virualenv using command:
` virtalenv venv`
- Activate virtualenv using command:
`source venv/bin/activate`
- Install project dependencies using command:
` pip install -r requirements.txt`

# To train the models
In this project, we have used DQN and DDQN agents to train our recommendation system
- To train DQN use command below:
`python src/train.py DQN`
- To train DDQN use command below:
`python src/train.py DDQN`

# To evaluate the models
In this project, we have used DQN and DDQN agents to train our recommendation system
- To evaluate DQN use command below:
`python src/evaluate.py DQN`
- To evaluate DDQN use command below:
`python src/evaluate.py DDQN`

# To generate recommendation for a single user
- To generate recommendation using DQN use command below:
`python src/recommend.py DQN [user_id]`
- To generate recommendation using DDQN use command below:
`python src/recommend.py DDQN [user_id]`
