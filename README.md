# Lane detection

Lane detection using CHEVP and B - snakes

### Prerequisites
Docker

### Installation
1. Download the project.
2. Position yourself in the folder lane_detection-master
3. Run: docker build -t lane .

### Running the tests
1. Run: docker run -ti -e DISPLAY=$DISPLAY -v $(pwd):/tmp lane python lane.py 

This will execute the algorithm on all .jpg images in folders ./test/outdoor and ./test/indoor. Results are saved in the folder ./test

### Dataset used
https://www.researchgate.net/project/Dataset-for-Lane-Detection
