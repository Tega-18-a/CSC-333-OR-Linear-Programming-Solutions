Project Title: Linear Programming Problems Using Python
Description:
This project solves various Linear Programming (LP) problems using the graphical method. Linear programming helps in achieving the best outcome (such as maximizing profit or minimizing cost) in mathematical models with linear constraints.

The project uses Python's NumPy and Matplotlib libraries to plot the constraints and visualize the feasible region. We then calculate the optimal solution based on the objective function (either maximizing or minimizing).

Objective:
To solve the following Linear Programming problems:

1. Maximizing Profit: Maximizing the profit from a factory that produces two products, given constraints on machine time and raw materials.
2. Minimizing Cost: Minimizing costs while satisfying given constraints on labor and material usage.
3. Optimal Resource Allocation: Solving for the best combination of resources to optimize production while considering various constraints.

How to Solve:
Define the Objective Function:

This is the function that needs to be either maximized or minimized. For example:
Maximize 
Z= 3x_1 + 4x_2
Z=3x_1 +4x_2
​for profit, or

Minimize 
C =6x_1 + 7x_2
C=6x_1+7x_2
for cost.

Set Up the Constraints:

Each problem has constraints like:
2x_1 + 3x_2 <= 12
x_1 + 2x_2 <= 8
x_1,x_2 >= 0

Graphical Representation:

Plot the constraints on a graph.
Identify the feasible region that satisfies all constraints.
Evaluate the objective function at the corner points of the feasible region.

Find the Optimal Solution:

The optimal solution is found by evaluating the objective function at each corner point and selecting the one that maximizes or minimizes the objective.
Instructions on How to Run the Python Code:
Install Python (>= 3.7): Make sure Python is installed on your system. If not, download and install it from the official Python website: python.org.

Install Required Libraries: You will need the following Python libraries:

''bash''
pip install matplotlib

Run the Python Script: Open the provided Python script (Awaritoma CSC333.py) in a Jupyter Notebook or your preferred Python IDE.
Execute the Code: Run each code cell to plot the constraints, display the feasible region, and determine the optimal solution.

Author:

Name: Awaritoma AchomaKoghene OgheneTega
Course:CSC333  Computational Science and Numerical Methods
Matric Number: VUG/CSC/22/7278