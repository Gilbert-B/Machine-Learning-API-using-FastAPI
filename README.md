# Machine Learning API using FastAPI
Develop a Machine Learning API (Application Programming Interface) using FastAPI.

[![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
[![MIT licensed](https://img.shields.io/badge/license-mit-blue?style=for-the-badge&logo=appveyor)](./LICENSE)
![Python](https://img.shields.io/badge/python-3.9-blue.svg)

## Introduction

In this project, Icreate an API that might be requested to interact with a ML model. This enables individuals and organizations keep their model architecture secret or to make their model available to users already having an API. By creating an API, and deploying it, a model can receive request using the internet protocol as presented by the illustration below.

![API illustration](https://lh3.googleusercontent.com/-qVJ4ZsbjsmH6CnYbojsAR4ImyHV8yxsFVinunH-pX7VCapGvufcXiPak6YVKIrj9ZdiCHwK5UFtQW8yuU5t83pz6fbqN1F2p74OWuT5dObCPnTBuCYr_P1mUg8arbP0WuEt7j_A)

**Source** : *The benefits of Machine Learning APIs - UbiOps*


## Description

<!-- 
[FastAPI](https://fastapi.tiangolo.com/) # 
-->

There is  a minimal API demo with [FastAPI](https://fastapi.tiangolo.com/), this will make sure that everything works correctly. Then, I will then create my own API, this will allow you to interact with a Machine Learning model, that is:
- Pass data through a request;
- Get the data in using the API;
- Apply the necessary processing;
- Submit the processed data to the ML model to make the predictions;
- Process the predictions obtained and return them as the API's response ot the input request.




## Setup

Install the required packages to be able to run the evaluation locally.

You need to have [`Python 3`](https://www.python.org/) on your system (**a Python version lower than 3.10**). Then you can clone this repo and being at the repo's `root :: repository_name> ...`  follow the steps below:

- Windows:
        
        python -m venv venv; venv\Scripts\activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

- Linux & MacOs:
        
        python3 -m venv venv; source venv/bin/activate; python -m pip install -q --upgrade pip; python -m pip install -qr requirements.txt  

The both long command-lines have a same structure, they pipe multiple commands using the symbol ` ; ` but you may manually execute them one after another.

1. **Create the Python's virtual environment** that isolates the required libraries of the project to avoid conflicts;
2. **Activate the Python's virtual environment** so that the Python kernel & libraries will be those of the isolated environment;
3. **Upgrade Pip, the installed libraries/packages manager** to have the up-to-date version that will work correctly;
4. **Install the required libraries/packages** listed in the `requirements.txt` file so that it will be allow to import them into the python's scripts and notebooks without any issue.

**NB:** For MacOs users, please install `Xcode` if you have an issue.

## Run FastAPI

- Run the demo apps (being at the repository root):
        
  FastAPI:
    
    - Demo

          uvicorn src.demo_01.api:app --reload 

    <!-- - Salary prediction

          uvicorn src.salary.api:app --reload  -->


  - Go to your browser at the following address, to explore the api's documentation :
        
      http://127.0.0.1:8000/docs


<!-- ## Screenshots

<table>
    <tr>
        <th>FastAPI</th>
        <th>FastAPI</th>
    </tr>
    <tr>
        <td><img src="./screenshots/.png"/></td>
        <td><img src="./screenshots/.png"/></td>
    </tr>
</table> -->


## Resources
Here are some ressources you would read to have a good understanding of FastAPI :
- [Tutorial - User Guide](https://fastapi.tiangolo.com/tutorial/)
- [Video - Building a Machine Learning API in 15 Minutes ](https://youtu.be/C82lT9cWQiA)
- [FastAPI for Machine Learning: Live coding an ML web application](https://www.youtube.com/watch?v=_BZGtifh_gw)
- [Video - Deploy ML models with FastAPI, Docker, and Heroku ](https://www.youtube.com/watch?v=h5wLuVDr0oc)
- [FastAPI Tutorial Series](https://www.youtube.com/watch?v=tKL6wEqbyNs&list=PLShTCj6cbon9gK9AbDSxZbas1F6b6C_Mx)
- [Http status codes](https://www.linkedin.com/feed/update/urn:li:activity:7017027658400063488?utm_source=share&utm_medium=member_desktop)





## Contributing

Feel free to make a PR or report an issue 😃.

Oh, one more thing, please do not forget to put a description when you make your PR 🙂.

## Author

- [Gilbert Botchway](https://www.linkedin.com/in/gilbert-botchway/)
