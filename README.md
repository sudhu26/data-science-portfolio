# Data science portfolio
---
Author: Tyler Peterson
- [LinkedIn](https://www.linkedin.com/in/petersontylerd/)
- [GitHub](https://github.com/petersontylerd)
- [DockerHub](https://hub.docker.com/u/petersontylerd)
- petersontylerd@gmail.com

This portfolio intends to serve as a tangible representation of my passion for continuously developing in the field of data science.

## Setup - use Docker
---
While you are certainly welcome to simply clone this repository and do what you want with it, I recommend using using the repository [docker-portfolio-vm](https://github.com/petersontylerd/docker-portfolio-vm), which I created to facilitate seamless interaction with my portfolio. 

[docker-portfolio-vm](https://github.com/petersontylerd/docker-portfolio-vm) leverages Docker to create an environment that contains all dependencies needed to execute the notebooks and scripts. The Dockerfile sets up a Python environment, clones this portfolio repository, and installs all required Python packages. The Dockerfile also utilizes Git LFS (large file storage), which helps to avoid downloading data files until necessary.

Three steps is all it takes to get up and running:

1. Get [docker-portfolio-vm](https://github.com/petersontylerd/docker-portfolio-vm):

```
git clone https://github.com/petersontylerd/docker-portfolio-vm.git
```

2. [docker-portfolio-vm](https://github.com/petersontylerd/docker-portfolio-vm) contains a docker-compose file, which makes it simple to create a container and launch a jupyter kernel for seamless exploration of the portfolio.

Create a container using docker-compose:

```
docker-compose up
```

This will also start the jupyter kernel. Notice the token in the URL - copy the alphanumerical string to the right of 'token=' in the URL

3. Go to [localhost:8888](localhost:8888) in your browser

In the text box next to 'Password or token:' at the top of the browser screen, paste in the token that was copied from the terminal.

And that's it.

## Portfolio Sections
---

### academics

Compilations of course work completed during Master of Data Science program at Indiana University.

  - [Applied Machine Learning](https://github.com/Petersontylerd/DataSciencePortfolio/tree/master/Academics/AppliedMachineLearning) - Class taken during Summer 2018 session. Directory includes my term project, all required homework assignments, as well as all optional assignments and lab sessions.


### kaggle

Kaggle competitions.

 - (_coming soon_)


### projects

Large-scale data science projects.

 - (_coming soon_)


### textbooks

Thorough dissection of textbooks and accompanying code sets. Emphasis placed on demonstrating mastery of key concepts by re-articulating key points and annotating, and often expanding upon, textbook code sets.

  - [Python Machine Learning, 2nd Edition](https://github.com/Petersontylerd/DataSciencePortfolio/tree/master/AnnotatedTextWalkthroughs/PythonMachineLearning2ndEd) - Walkthrough of Sebastian Raschka's [textbook](https://www.oreilly.com/library/view/python-machine-learning/9781787125933/). 

  - [Deep Learning with PyTorch](https://github.com/petersontylerd/data-science-portfolio/tree/master/textbooks/DeepLearningWithPyTorch) - Walkthrough of Vishnu Subramanian's PyTorch [textbook](https://www.oreilly.com/library/view/deep-learning-with/9781788624336/). 


### tutorials

Like most data science enthusiasts, I scour the internet for interesting tutorials. Here is a collection of some of my recent favorites.

 - (_coming soon_)

