import numpy as np
import pandas as pd
import json
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Enums for system states
class RobotStatus(Enum):
    OPERATIONAL = "operational"
    WARNING = "warning"
    CRITICAL = "critical"
    MAINTENANCE = "maintenance"
    OFFLINE = "offline"

class MaintenanceType(Enum):
    PREVENTIVE = "preventive"
    PREDICTIVE = "predictive"
    CORRECTIVE = "corrective"
    EMERGENCY = "emergency"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4

# Data structures
@dataclass
class SensorReading:
    robot_id: str
    timestamp: datetime
    temperature: float
    vibration: float
    pressure: float
    current: float
    voltage: float
    speed: float
    load: float
    operating_hours: float

@dataclass
class MaintenanceTask:
    task_id: str
    robot_id: str
    task_type: MaintenanceType
    priority: Priority
    estimated_duration: int  # minutes
    required_parts: List[str]
    scheduled_time: datetime
    completion_time: Optional[datetime] = None
    status: str = "pending"
    description: str = ""

@dataclass
class PredictionResult:
    robot_id: str
    failure_probability: float
    predicted_failure_time: Optional[datetime]
    recommended_action: str
    confidence: float
    anomaly_score: float

class IoTSensorSimulator:
    """Simulates IoT sensor data for industrial robots"""
    
    def __init__(self):
        self.robots = {}
        self.base_params = {
            'temperature': (65, 85),  # Normal range
            'vibration': (0.1, 0.3),
            'pressure': (120, 140),
            'current': (8, 12),
            'voltage': (220, 240),
            'speed': (80, 120),
            'load': (40, 80),
        }
    
    def register_robot(self, robot_id: str, status: RobotStatus = RobotStatus.OPERATIONAL):
        """Register a new robot in the system"""
        self.robots[robot_id] = {
            'status': status,
            'operating_hours': np.random.uniform(0, 8760),  # 0-1 year
            'last_maintenance': datetime.now() - timedelta(days=np.random.randint(0, 180)),
            'degradation_factor': np.random.uniform(0.8, 1.2)
        }
    
    def generate_sensor_data(self, robot_id: str) -> SensorReading:
        """Generate realistic sensor data with degradation patterns"""
        if robot_id not in self.robots:
            raise ValueError(f"Robot {robot_id} not registered")
        
        robot = self.robots[robot_id]
        degradation = robot['degradation_factor']
        operating_hours = robot['operating_hours']
        
        # Simulate degradation over time
        age_factor = min(1.0 + (operating_hours / 8760) * 0.2, 1.5)
        
        # Generate base readings
        temp = np.random.normal(75, 5) * degradation * age_factor
        vibration = np.random.normal(0.2, 0.05) * degradation * age_factor
        pressure = np.random.normal(130, 5) * (2 - degradation) * age_factor
        current = np.random.normal(10, 1) * degradation * age_factor
        voltage = np.random.normal(230, 5) * (2 - degradation)
        speed = np.random.normal(100, 10) * (2 - degradation)
        load = np.random.normal(60, 10) * degradation * age_factor
        
        # Add anomalies based on robot status
        if robot['status'] == RobotStatus.WARNING:
            temp *= 1.1
            vibration *= 1.2
            current *= 1.1
        elif robot['status'] == RobotStatus.CRITICAL:
            temp *= 1.2
            vibration *= 1.5
            current *= 1.3
            pressure *= 0.8
        
        # Update operating hours
        robot['operating_hours'] += 0.1  # Increment by 6 minutes
        
        return SensorReading(
            robot_id=robot_id,
            timestamp=datetime.now(),
            temperature=temp,
            vibration=vibration,
            pressure=pressure,
            current=current,
            voltage=voltage,
            speed=speed,
            load=load,
            operating_hours=operating_hours
        )

class PredictiveMaintenanceEngine:
    """Core ML engine for predictive maintenance"""
    
    def __init__(self):
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.failure_predictor = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = ['temperature', 'vibration', 'pressure', 'current', 
                               'voltage', 'speed', 'load', 'operating_hours']
    
    def prepare_features(self, sensor_data: List[SensorReading]) -> np.ndarray:
        """Convert sensor readings to feature matrix"""
        features = []
        for reading in sensor_data:
            features.append([
                reading.temperature,
                reading.vibration,
                reading.pressure,
                reading.current,
                reading.voltage,
                reading.speed,
                reading.load,
                reading.operating_hours
            ])
        return np.array(features)
    
    def train_models(self, historical_data: List[SensorReading], failure_labels: List[int]):
        """Train the predictive models"""
        X = self.prepare_features(historical_data)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train anomaly detector
        self.anomaly_detector.fit(X_scaled)
        
        # Train failure predictor
        if len(failure_labels) == len(historical_data):
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, failure_labels, test_size=0.2, random_state=42
            )
            self.failure_predictor.fit(X_train, y_train)
            
            # Evaluate model
            y_pred = self.failure_predictor.predict(X_test)
            logger.info(f"Model Performance:\n{classification_report(y_test, y_pred)}")
        
        self.is_trained = True
        logger.info("Models trained successfully")
    
    def predict_failure(self, sensor_reading: SensorReading) -> PredictionResult:
        """Predict failure probability for a single sensor reading"""
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        features = self.prepare_features([sensor_reading])
        features_scaled = self.scaler.transform(features)
        
        # Get anomaly score
        anomaly_score = self.anomaly_detector.decision_function(features_scaled)[0]
        is_anomaly = self.anomaly_detector.predict(features_scaled)[0] == -1
        
        # Get failure probability
        failure_prob = self.failure_predictor.predict_proba(features_scaled)[0][1]
        
        # Calculate confidence
        confidence = max(abs(anomaly_score), failure_prob)
        
        # Determine recommended action
        if failure_prob > 0.8 or anomaly_score < -0.5:
            action = "Immediate maintenance required"
            predicted_failure_time = datetime.now() + timedelta(hours=2)
        elif failure_prob > 0.6 or anomaly_score < -0.3:
            action = "Schedule maintenance within 24 hours"
            predicted_failure_time = datetime.now() + timedelta(hours=12)
        elif failure_prob > 0.4 or anomaly_score < -0.1:
            action = "Monitor closely, schedule preventive maintenance"
            predicted_failure_time = datetime.now() + timedelta(days=3)
        else:
            action = "Normal operation"
            predicted_failure_time = None
        
        return PredictionResult(
            robot_id=sensor_reading.robot_id,
            failure_probability=failure_prob,
            predicted_failure_time=predicted_failure_time,
            recommended_action=action,
            confidence=confidence,
            anomaly_score=anomaly_score
        )

class MaintenanceScheduler:
    """Intelligent maintenance task scheduler"""
    
    def __init__(self):
        self.tasks = []
        self.technicians = []
        self.parts_inventory = {}
    
    def add_technician(self, technician_id: str, skills: List[str]):
        """Add a maintenance technician"""
        self.technicians.append({
            'id': technician_id,
            'skills': skills,
            'current_task': None,
            'available_time': datetime.now()
        })
    
    def update_inventory(self, part_name: str, quantity: int):
        """Update parts inventory"""
        self.parts_inventory[part_name] = quantity
    
    def create_maintenance_task(self, prediction: PredictionResult) -> MaintenanceTask:
        """Create a maintenance task based on prediction"""
        task_id = f"TASK_{prediction.robot_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Determine task type and priority
        if prediction.failure_probability > 0.8:
            task_type = MaintenanceType.EMERGENCY
            priority = Priority.CRITICAL
            duration = 120  # 2 hours
            parts = ['emergency_repair_kit', 'replacement_parts']
        elif prediction.failure_probability > 0.6:
            task_type = MaintenanceType.PREDICTIVE
            priority = Priority.HIGH
            duration = 90
            parts = ['sensors', 'lubricants', 'filters']
        elif prediction.failure_probability > 0.4:
            task_type = MaintenanceType.PREVENTIVE
            priority = Priority.MEDIUM
            duration = 60
            parts = ['lubricants', 'filters']
        else:
            task_type = MaintenanceType.PREVENTIVE
            priority = Priority.LOW
            duration = 30
            parts = ['inspection_tools']
        
        # Schedule time based on priority
        if priority == Priority.CRITICAL:
            scheduled_time = datetime.now() + timedelta(minutes=30)
        elif priority == Priority.HIGH:
            scheduled_time = datetime.now() + timedelta(hours=2)
        elif priority == Priority.MEDIUM:
            scheduled_time = datetime.now() + timedelta(hours=8)
        else:
            scheduled_time = datetime.now() + timedelta(days=1)
        
        task = MaintenanceTask(
            task_id=task_id,
            robot_id=prediction.robot_id,
            task_type=task_type,
            priority=priority,
            estimated_duration=duration,
            required_parts=parts,
            scheduled_time=scheduled_time,
            description=prediction.recommended_action
        )
        
        self.tasks.append(task)
        return task
    
    def optimize_schedule(self) -> List[MaintenanceTask]:
        """Optimize maintenance schedule using priority and resource availability"""
        # Sort tasks by priority and scheduled time
        sorted_tasks = sorted(self.tasks, 
                            key=lambda x: (x.priority.value, x.scheduled_time), 
                            reverse=True)
        
        optimized_schedule = []
        current_time = datetime.now()
        
        for task in sorted_tasks:
            if task.status == "pending":
                # Check parts availability
                parts_available = all(
                    self.parts_inventory.get(part, 0) > 0 
                    for part in task.required_parts
                )
                
                if parts_available:
                    # Find available technician
                    available_tech = next(
                        (tech for tech in self.technicians 
                         if tech['available_time'] <= current_time), 
                        None
                    )
                    
                    if available_tech:
                        # Schedule the task
                        task.scheduled_time = max(current_time, task.scheduled_time)
                        available_tech['available_time'] = (
                            task.scheduled_time + timedelta(minutes=task.estimated_duration)
                        )
                        optimized_schedule.append(task)
                        
                        # Update inventory
                        for part in task.required_parts:
                            self.parts_inventory[part] -= 1
        
        return optimized_schedule

class MultiAgentCoordinator:
    """Coordinates multiple agents in the maintenance system"""
    
    def __init__(self):
        self.agents = {}
        self.message_queue = asyncio.Queue()
        self.is_running = False
    
    def register_agent(self, agent_id: str, agent_type: str, capabilities: List[str]):
        """Register an agent in the system"""
        self.agents[agent_id] = {
            'type': agent_type,
            'capabilities': capabilities,
            'status': 'active',
            'last_activity': datetime.now()
        }
    
    async def send_message(self, from_agent: str, to_agent: str, message: Dict):
        """Send message between agents"""
        await self.message_queue.put({
            'from': from_agent,
            'to': to_agent,
            'message': message,
            'timestamp': datetime.now()
        })
    
    async def process_messages(self):
        """Process messages in the queue"""
        while self.is_running:
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                logger.info(f"Message from {message['from']} to {message['to']}: {message['message']}")
                # Process message based on agent types and capabilities
                await self.handle_message(message)
            except asyncio.TimeoutError:
                continue
    
    async def handle_message(self, message: Dict):
        """Handle different types of messages"""
        msg_type = message['message'].get('type')
        
        if msg_type == 'maintenance_request':
            # Forward to maintenance scheduler
            logger.info(f"Processing maintenance request for robot {message['message']['robot_id']}")
        elif msg_type == 'sensor_alert':
            # Forward to monitoring agent
            logger.info(f"Processing sensor alert: {message['message']['alert']}")
        elif msg_type == 'task_completed':
            # Update task status
            logger.info(f"Task {message['message']['task_id']} completed")

class PredictiveMaintenanceSystem:
    """Main system orchestrator"""
    
    def __init__(self):
        self.sensor_simulator = IoTSensorSimulator()
        self.ml_engine = PredictiveMaintenanceEngine()
        self.scheduler = MaintenanceScheduler()
        self.coordinator = MultiAgentCoordinator()
        self.db_connection = None
        self.is_running = False
        
        # Initialize database
        self.init_database()
        
        # Register agents
        self.coordinator.register_agent("sensor_agent", "monitoring", ["data_collection", "alert_generation"])
        self.coordinator.register_agent("prediction_agent", "analysis", ["failure_prediction", "anomaly_detection"])
        self.coordinator.register_agent("scheduler_agent", "planning", ["task_scheduling", "resource_optimization"])
    
    def init_database(self):
        """Initialize SQLite database for storing system data"""
        self.db_connection = sqlite3.connect('maintenance_system.db')
        cursor = self.db_connection.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sensor_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                robot_id TEXT,
                timestamp DATETIME,
                temperature REAL,
                vibration REAL,
                pressure REAL,
                current REAL,
                voltage REAL,
                speed REAL,
                load REAL,
                operating_hours REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maintenance_tasks (
                task_id TEXT PRIMARY KEY,
                robot_id TEXT,
                task_type TEXT,
                priority INTEGER,
                estimated_duration INTEGER,
                required_parts TEXT,
                scheduled_time DATETIME,
                completion_time DATETIME,
                status TEXT,
                description TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                robot_id TEXT,
                timestamp DATETIME,
                failure_probability REAL,
                predicted_failure_time DATETIME,
                recommended_action TEXT,
                confidence REAL,
                anomaly_score REAL
            )
        ''')
        
        self.db_connection.commit()
    
    def add_robot(self, robot_id: str):
        """Add a new robot to the system"""
        self.sensor_simulator.register_robot(robot_id)
        logger.info(f"Robot {robot_id} added to system")
    
    def add_technician(self, technician_id: str, skills: List[str]):
        """Add a maintenance technician"""
        self.scheduler.add_technician(technician_id, skills)
        logger.info(f"Technician {technician_id} added with skills: {skills}")
    
    def update_inventory(self, part_name: str, quantity: int):
        """Update parts inventory"""
        self.scheduler.update_inventory(part_name, quantity)
        logger.info(f"Updated inventory: {part_name} = {quantity}")
    
    def store_sensor_reading(self, reading: SensorReading):
        """Store sensor reading in database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO sensor_readings 
            (robot_id, timestamp, temperature, vibration, pressure, current, voltage, speed, load, operating_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            reading.robot_id, reading.timestamp, reading.temperature, reading.vibration,
            reading.pressure, reading.current, reading.voltage, reading.speed,
            reading.load, reading.operating_hours
        ))
        self.db_connection.commit()
    
    def store_prediction(self, prediction: PredictionResult):
        """Store prediction result in database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT INTO predictions 
            (robot_id, timestamp, failure_probability, predicted_failure_time, recommended_action, confidence, anomaly_score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            prediction.robot_id, datetime.now(), prediction.failure_probability,
            prediction.predicted_failure_time, prediction.recommended_action,
            prediction.confidence, prediction.anomaly_score
        ))
        self.db_connection.commit()
    
    def store_maintenance_task(self, task: MaintenanceTask):
        """Store maintenance task in database"""
        cursor = self.db_connection.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO maintenance_tasks 
            (task_id, robot_id, task_type, priority, estimated_duration, required_parts, 
             scheduled_time, completion_time, status, description)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            task.task_id, task.robot_id, task.task_type.value, task.priority.value,
            task.estimated_duration, json.dumps(task.required_parts), task.scheduled_time,
            task.completion_time, task.status, task.description
        ))
        self.db_connection.commit()
    
    def generate_training_data(self, robot_ids: List[str], num_samples: int = 1000):
        """Generate training data for ML models"""
        training_data = []
        labels = []
        
        for _ in range(num_samples):
            robot_id = np.random.choice(robot_ids)
            
            # Simulate different robot conditions
            if np.random.random() < 0.1:  # 10% failure cases
                self.sensor_simulator.robots[robot_id]['status'] = RobotStatus.CRITICAL
                labels.append(1)
            else:
                self.sensor_simulator.robots[robot_id]['status'] = RobotStatus.OPERATIONAL
                labels.append(0)
            
            reading = self.sensor_simulator.generate_sensor_data(robot_id)
            training_data.append(reading)
            
            # Reset robot status
            self.sensor_simulator.robots[robot_id]['status'] = RobotStatus.OPERATIONAL
        
        return training_data, labels
    
    def train_system(self, robot_ids: List[str]):
        """Train the ML models"""
        logger.info("Generating training data...")
        training_data, labels = self.generate_training_data(robot_ids)
        
        logger.info("Training ML models...")
        self.ml_engine.train_models(training_data, labels)
        
        logger.info("System training completed")
    
    async def run_monitoring_cycle(self):
        """Run one monitoring cycle for all robots"""
        for robot_id in self.sensor_simulator.robots.keys():
            # Get sensor reading
            reading = self.sensor_simulator.generate_sensor_data(robot_id)
            self.store_sensor_reading(reading)
            
            # Make prediction
            if self.ml_engine.is_trained:
                prediction = self.ml_engine.predict_failure(reading)
                self.store_prediction(prediction)
                
                # Create maintenance task if needed
                if prediction.failure_probability > 0.4:
                    task = self.scheduler.create_maintenance_task(prediction)
                    self.store_maintenance_task(task)
                    
                    # Send alert through agent system
                    await self.coordinator.send_message(
                        "sensor_agent", 
                        "prediction_agent",
                        {
                            'type': 'maintenance_request',
                            'robot_id': robot_id,
                            'priority': task.priority.value,
                            'prediction': asdict(prediction)
                        }
                    )
                
                logger.info(f"Robot {robot_id}: Failure probability = {prediction.failure_probability:.2f}, "
                           f"Anomaly score = {prediction.anomaly_score:.2f}")
    
    async def start_system(self):
        """Start the predictive maintenance system"""
        self.is_running = True
        self.coordinator.is_running = True
        
        logger.info("Starting Predictive Maintenance System...")
        
        # Start message processing
        message_task = asyncio.create_task(self.coordinator.process_messages())
        
        # Main monitoring loop
        while self.is_running:
            await self.run_monitoring_cycle()
            await asyncio.sleep(5)  # 5-second monitoring interval
        
        # Cleanup
        message_task.cancel()
        self.coordinator.is_running = False
    
    def stop_system(self):
        """Stop the system"""
        self.is_running = False
        logger.info("System stopped")
    
    def close_database(self):
        """Close database connection"""
        if self.db_connection:
            self.db_connection.close()
            self.db_connection = None
    
    def get_system_status(self) -> Dict:
        """Get current system status"""
        if not self.db_connection:
            logger.warning("Database connection is closed")
            return {
                'robots': {},
                'pending_tasks': 0,
                'system_uptime': datetime.now(),
                'agents_active': len(self.coordinator.agents)
            }
        
        try:
            cursor = self.db_connection.cursor()
            
            # Get robot status
            robot_status = {}
            for robot_id in self.sensor_simulator.robots.keys():
                cursor.execute('''
                    SELECT failure_probability, anomaly_score 
                    FROM predictions 
                    WHERE robot_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (robot_id,))
                
                result = cursor.fetchone()
                if result:
                    robot_status[robot_id] = {
                        'failure_probability': result[0],
                        'anomaly_score': result[1],
                        'status': self.sensor_simulator.robots[robot_id]['status'].value
                    }
            
            # Get pending tasks
            cursor.execute('''
                SELECT COUNT(*) FROM maintenance_tasks WHERE status = 'pending'
            ''')
            pending_tasks = cursor.fetchone()[0]
            
            return {
                'robots': robot_status,
                'pending_tasks': pending_tasks,
                'system_uptime': datetime.now(),
                'agents_active': len(self.coordinator.agents)
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {
                'robots': {},
                'pending_tasks': 0,
                'system_uptime': datetime.now(),
                'agents_active': len(self.coordinator.agents)
            }

# Example usage and demo
async def main():
    """Main function demonstrating the system"""
    print("="*80)
    print("AI-POWERED PREDICTIVE MAINTENANCE SYSTEM FOR INDUSTRIAL ROBOTICS")
    print("="*80)
    
    system = PredictiveMaintenanceSystem()
    
    # Add robots
    robot_ids = ['ROBOT_001', 'ROBOT_002', 'ROBOT_003', 'ROBOT_004']
    print("\n🤖 INITIALIZING ROBOTS:")
    for robot_id in robot_ids:
        system.add_robot(robot_id)
        print(f"  ✓ {robot_id} registered and online")
    
    # Add technicians
    print("\n👨‍🔧 ADDING MAINTENANCE TECHNICIANS:")
    technicians = [
        ('TECH_001', ['electrical', 'mechanical']),
        ('TECH_002', ['hydraulic', 'pneumatic']),
        ('TECH_003', ['software', 'sensors'])
    ]
    for tech_id, skills in technicians:
        system.add_technician(tech_id, skills)
        print(f"  ✓ {tech_id} added with skills: {', '.join(skills)}")
    
    # Update inventory
    print("\n📦 UPDATING PARTS INVENTORY:")
    parts = ['sensors', 'lubricants', 'filters', 'emergency_repair_kit', 'replacement_parts', 'inspection_tools']
    for part in parts:
        system.update_inventory(part, 10)
        print(f"  ✓ {part}: 10 units in stock")
    
    # Train the system
    print("\n🧠 TRAINING AI MODELS:")
    print("  • Generating synthetic training data...")
    print("  • Training anomaly detection model...")
    print("  • Training failure prediction model...")
    system.train_system(robot_ids)
    print("  ✓ AI models trained successfully!")
    
    # Show initial robot status
    print("\n📊 INITIAL ROBOT STATUS:")
    for robot_id in robot_ids:
        robot_info = system.sensor_simulator.robots[robot_id]
        print(f"  {robot_id}:")
        print(f"    Status: {robot_info['status'].value}")
        print(f"    Operating Hours: {robot_info['operating_hours']:.1f}")
        print(f"    Last Maintenance: {robot_info['last_maintenance'].strftime('%Y-%m-%d %H:%M')}")
    
    # Run monitoring cycles manually to show detailed output
    print("\n🔍 STARTING REAL-TIME MONITORING:")
    print("="*50)
    
    for cycle in range(8):  # Run 8 monitoring cycles
        print(f"\n--- MONITORING CYCLE {cycle + 1} ---")
        
        for robot_id in robot_ids:
            # Get sensor reading
            reading = system.sensor_simulator.generate_sensor_data(robot_id)
            system.store_sensor_reading(reading)
            
            # Display sensor data
            print(f"\n📈 {robot_id} SENSOR DATA:")
            print(f"  Temperature: {reading.temperature:.1f}°C")
            print(f"  Vibration: {reading.vibration:.3f} G")
            print(f"  Pressure: {reading.pressure:.1f} PSI")
            print(f"  Current: {reading.current:.1f} A")
            print(f"  Voltage: {reading.voltage:.1f} V")
            print(f"  Speed: {reading.speed:.1f} RPM")
            print(f"  Load: {reading.load:.1f}%")
            
            # Make prediction
            if system.ml_engine.is_trained:
                prediction = system.ml_engine.predict_failure(reading)
                system.store_prediction(prediction)
                
                # Display prediction
                print(f"🔮 AI PREDICTION:")
                print(f"  Failure Probability: {prediction.failure_probability:.1%}")
                print(f"  Anomaly Score: {prediction.anomaly_score:.3f}")
                print(f"  Confidence: {prediction.confidence:.1%}")
                print(f"  Recommended Action: {prediction.recommended_action}")
                
                # Create maintenance task if needed
                if prediction.failure_probability > 0.4:
                    task = system.scheduler.create_maintenance_task(prediction)
                    system.store_maintenance_task(task)
                    
                    print(f"⚠️  MAINTENANCE TASK CREATED:")
                    print(f"  Task ID: {task.task_id}")
                    print(f"  Type: {task.task_type.value}")
                    print(f"  Priority: {task.priority.name}")
                    print(f"  Estimated Duration: {task.estimated_duration} minutes")
                    print(f"  Required Parts: {', '.join(task.required_parts)}")
                    print(f"  Scheduled Time: {task.scheduled_time.strftime('%Y-%m-%d %H:%M')}")
                    
                    # Send alert through agent system
                    await system.coordinator.send_message(
                        "sensor_agent", 
                        "prediction_agent",
                        {
                            'type': 'maintenance_request',
                            'robot_id': robot_id,
                            'priority': task.priority.value,
                            'prediction': asdict(prediction)
                        }
                    )
                
                # Add some status indicators
                if prediction.failure_probability > 0.8:
                    print("  🔴 STATUS: CRITICAL - IMMEDIATE ACTION REQUIRED")
                elif prediction.failure_probability > 0.6:
                    print("  🟡 STATUS: WARNING - SCHEDULE MAINTENANCE SOON")
                elif prediction.failure_probability > 0.4:
                    print("  🟠 STATUS: CAUTION - MONITOR CLOSELY")
                else:
                    print("  🟢 STATUS: NORMAL OPERATION")
        
        # Simulate some robot degradation
        if cycle == 3:  # After 3 cycles, degrade one robot
            system.sensor_simulator.robots['ROBOT_002']['status'] = RobotStatus.WARNING
            system.sensor_simulator.robots['ROBOT_002']['degradation_factor'] = 1.3
            print(f"\n⚠️  ROBOT_002 showing signs of wear - degradation factor increased")
        
        if cycle == 5:  # After 5 cycles, another robot becomes critical
            system.sensor_simulator.robots['ROBOT_003']['status'] = RobotStatus.CRITICAL
            system.sensor_simulator.robots['ROBOT_003']['degradation_factor'] = 1.6
            print(f"\n🚨 ROBOT_003 entering critical condition!")
        
        await asyncio.sleep(2)  # 2 second delay between cycles
    
    # Optimize maintenance schedule
    print("\n🗓️  OPTIMIZING MAINTENANCE SCHEDULE:")
    print("="*40)
    optimized_tasks = system.scheduler.optimize_schedule()
    
    if optimized_tasks:
        print(f"Optimized schedule created with {len(optimized_tasks)} tasks:")
        for i, task in enumerate(optimized_tasks, 1):
            print(f"\n  {i}. Task {task.task_id}:")
            print(f"     Robot: {task.robot_id}")
            print(f"     Type: {task.task_type.value}")
            print(f"     Priority: {task.priority.name}")
            print(f"     Scheduled: {task.scheduled_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"     Duration: {task.estimated_duration} minutes")
            print(f"     Parts: {', '.join(task.required_parts)}")
    else:
        print("No maintenance tasks currently scheduled.")
    
    # Show final system status
    print("\n📊 FINAL SYSTEM STATUS:")
    print("="*30)
    status = system.get_system_status()
    
    print(f"System Uptime: {status['system_uptime'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Active Agents: {status['agents_active']}")
    print(f"Pending Tasks: {status['pending_tasks']}")
    
    if status['robots']:
        print("\nRobot Health Summary:")
        for robot_id, robot_status in status['robots'].items():
            failure_prob = robot_status.get('failure_probability', 0)
            anomaly_score = robot_status.get('anomaly_score', 0)
            current_status = robot_status.get('status', 'unknown')
            
            if failure_prob > 0.8:
                health_icon = "🔴"
            elif failure_prob > 0.6:
                health_icon = "🟡"
            elif failure_prob > 0.4:
                health_icon = "🟠"
            else:
                health_icon = "🟢"
                
            print(f"  {health_icon} {robot_id}: {failure_prob:.1%} failure risk ({current_status})")
    else:
        print("\nNo robot status data available")
    
    # Show inventory status
    print(f"\nParts Inventory Status:")
    for part, quantity in system.scheduler.parts_inventory.items():
        if quantity < 3:
            print(f"  🔴 {part}: {quantity} units (LOW STOCK)")
        elif quantity < 5:
            print(f"  🟡 {part}: {quantity} units")
        else:
            print(f"  🟢 {part}: {quantity} units")
    
    # Stop the system and close database
    system.stop_system()
    system.close_database()
    
    print("\n" + "="*80)
    print("SYSTEM DEMONSTRATION COMPLETED")
    print("="*80)

if __name__ == "__main__":
    asyncio.run(main())