# Dynamic-Reconfiguration-for-Manufacturing-Lines
This project presents the development of a digital-twin based reconfiguration system for disturbance handling in manufacturing lines. The system consists of four modules, the Ontology Model, the Digital Twin, the Simulator, and the Optimizer. 
The ontology model defines the manufacturing line's objects, including their properties, functionalities, and relationships. Objects can be Workers, Robots, and Machines. The objects and their relationships are stored in an excel file, which is read by the code to construct the corrsponding python classes.
The Digital Twin is responsible to detect disturbance events that cause deviation of the expected production performance. 
The Simulator models real time production with Descrete Event Simulation. 
The Optimizer is responsible to optimize the current manufucturing line to restore the production efficiently, and is activated by the Digital Twin when a disturbance event is detected.     
