# Lane detection

Lane detection using CHEVP and B - snakes

### Prerequisites
Docker

### Running the tests
1. Download the project.
2. Position yourself in the folder lane_detection
3. Run: docker build -t lane .
4. Run: docker run -ti -e DISPLAY=$DISPLAY -v $(pwd):/tmp lane python lane.py 

This will execute the algorithm on all .jpg images in folders ./test/outdoor and ./test/indoor. Results are saved in the folder ./test
