import json
import numpy as np
import random
from pyswarm import pso
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')


# Constants
DAYS = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday"]
TIMESLOTS = [
    "6:40-8:20", "8:20-10:00", "10:00-10:50 (Break)", "10:50-12:30", "12:30-14:10",
    "14:10-15:00 (Break)", "15:00-16:40"
]
TOTAL_PERIODS = 7  # Number of periods per day

# Load or create timetable data
def load_or_create_data():
    try:
        with open('timetable_data.json', 'r') as file:
            data = json.load(file)
    except FileNotFoundError:
        data = {
            "instructors_bct": [],
            "subjects_bct": [],
            "instructor_type_bct": [],
            "labs_bct": [],
            "instructors_bei": [],
            "subjects_bei": [],
            "instructor_type_bei": [],
            "labs_bei": []
        }
        
        print("Enter BCT instructor details:")
        num_bct = int(input("Enter the number of BCT instructors: "))
        for _ in range(num_bct):
            name = input("Enter instructor name: ")
            subject = input("Enter subject: ")
            instructor_type = input("Enter instructor type (f for full-time, p for part-time): ")
            lab = input("Does this instructor have a lab session? (y/n): ").lower() == 'y'
            
            data['instructors_bct'].append(name)
            data['subjects_bct'].append(subject)
            data['instructor_type_bct'].append(instructor_type)
            if lab:
                data['labs_bct'].append(subject)

        print("Enter BEI instructor details:")
        num_bei = int(input("Enter the number of BEI instructors: "))
        for _ in range(num_bei):
            name = input("Enter instructor name: ")
            subject = input("Enter subject: ")
            instructor_type = input("Enter instructor type (f for full-time, p for part-time): ")
            lab = input("Does this instructor have a lab session? (y/n): ").lower() == 'y'
            
            data['instructors_bei'].append(name)
            data['subjects_bei'].append(subject)
            data['instructor_type_bei'].append(instructor_type)
            if lab:
                data['labs_bei'].append(subject)

        with open('timetable_data.json', 'w') as file:
            json.dump(data, file)
    
    return data

# Objective function for PSO
def objective_function(x, num_classes, num_instructors_bct, num_instructors_bei, num_periods, labs_bct, labs_bei, instructor_type_bct, instructor_type_bei, subjects_bct, subjects_bei):
    penalty = 0

    num_periods_per_day = TOTAL_PERIODS
    num_days = len(DAYS)
    schedule = np.reshape(x, (num_periods_per_day, num_days))

    instructor_day_count = [0] * (num_instructors_bct + num_instructors_bei)
    subject_day_count = {}

    for period in range(num_periods_per_day):
        for day in range(num_days):
            if period == 2 or period == 5:
                # It's a break period
                if schedule[period, day] != -1:
                    penalty += 100
                continue

            instructor = int(schedule[period, day])
            if instructor >= 0 and instructor < num_instructors_bct:
                subject = subjects_bct[instructor]
                if instructor_type_bct[instructor] == 'p' and period > 2:
                    penalty += 50
                if instructor_type_bct[instructor] == 'f' and period <= 2:
                    penalty += 50

            elif instructor >= num_instructors_bct and instructor < num_instructors_bct + num_instructors_bei:
                subject = subjects_bei[instructor - num_instructors_bct]
                if instructor_type_bei[instructor - num_instructors_bct] == 'p' and period > 2:
                    penalty += 50
                if instructor_type_bei[instructor - num_instructors_bct] == 'f' and period <= 2:
                    penalty += 50
            else:
                penalty += 100
                continue

            # Check if instructor has more than MAX_CLASSES_PER_DAY classes
            instructor_day_count[instructor] += 1
            if instructor_day_count[instructor] > 2:
                penalty += 50

            # Check for subject repetition on the same day
            if (day, subject) in subject_day_count:
                subject_day_count[(day, subject)] += 1
                penalty += 50
            else:
                subject_day_count[(day, subject)] = 1

            # Check for overlap with labs


    return penalty

def objective_function_with_tracking(x, num_classes, num_instructors_bct, num_instructors_bei, num_periods, labs_bct, labs_bei, instructor_type_bct, instructor_type_bei, subjects_bct, subjects_bei, penalties_history,infeasible_count):
    threshold=100
    penalty = objective_function(x, num_classes, num_instructors_bct, num_instructors_bei, num_periods, labs_bct, labs_bei, instructor_type_bct, instructor_type_bei, subjects_bct, subjects_bei)
    penalties_history.append(penalty)  # Track the penalty at this iteration
   # Track infeasibility based on penalty exceeding threshold
    if penalty > threshold:
        infeasible_count.append(1)  # Mark this iteration as infeasible
    else:
        infeasible_count.append(0)  # Feasible solution
    
    print(f"Iteration {len(penalties_history)}: Penalty = {penalty}")
    return penalty

# Generate timetable using PSO
def generate_timetable(num_classes, num_instructors_bct, num_instructors_bei, subjects_bct, subjects_bei, instructor_type_bct, instructor_type_bei, labs_bct, labs_bei):
    num_periods_per_day = TOTAL_PERIODS
    num_days = len(DAYS)
    num_periods = num_days * num_periods_per_day  # Total periods

    lb = [0] * num_periods
    ub = [num_instructors_bct + num_instructors_bei - 1] * num_periods

    penalties_history = []
    infeasible_count=[]
    avg_penalties = []
    best_penalties = []

    threshold = 100 

    def custom_objective_function(x, *args):
        penalty = objective_function_with_tracking(
            x, *args, penalties_history,infeasible_count
        )
        avg_penalty = np.mean(penalties_history[-100:])  # Compute average of the last 100 penalties
        best_penalty = np.min(penalties_history)  # Track the minimum penalty encountered

        avg_penalties.append(avg_penalty)
        best_penalties.append(best_penalty)
        
        return penalty


    xopt, fopt = pso(
        custom_objective_function, lb, ub,
        args=(num_classes, num_instructors_bct, num_instructors_bei, num_periods, labs_bct, labs_bei, instructor_type_bct, instructor_type_bei, subjects_bct, subjects_bei),
        swarmsize=50, maxiter=500,
        omega=0.5,   # Set the inertia weight (w1)
        phip=1.5,    # Set the cognitive coefficient (c1)
        phig=1.5    # Set the social coefficient (c2)
)
    


    # Penalty over iterations (convergence plot)
    plt.plot(avg_penalties, label='Penalty Over Iterations', color='r',linewidth=1,linestyle='--')
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Penalty', fontsize=12)
    plt.title('Penalty Convergence (Penalty vs Iteration)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig('convergence_plot.png', dpi=300)
    plt.close()

# After generating the timetable using PSO and tracking penalties

    # Plot the performance (best vs avg penalty)
    plt.plot(best_penalties, label='Best Penalty', color='b', linewidth=1)
    plt.plot(avg_penalties, label='Average Penalty', color='g', linewidth=1, linestyle='--')  # Dashed line for avg
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Penalty', fontsize=12)
    plt.title('PSO Performance (Best vs Average Penalty)', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)  # Dashed grid lines for better readability
    plt.tight_layout()  # Adjust layout to ensure everything fits
    plt.savefig('pso_performance.png', dpi=300)  # Save the plot with high resolution
    plt.close()  # Close the plot to prevent memory issues

    # 3. Infeasible Solutions Over Iterations
    plt.plot(infeasible_count, label='Infeasible Solutions', color='orange', linewidth=1)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Infeasible Solutions', fontsize=12)
    plt.title('Infeasible Solutions Over Iterations', fontsize=14)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('infeasibility_plot.png', dpi=300)  # Save infeasibility plot
    plt.close()  # Close the plot

    # Reshape according to (num_periods_per_day, num_days)
    return np.reshape(xopt, (num_periods_per_day, num_days))

    

# Display the timetable
def display_timetable(timetable, num_classes, class_names, instructors_bct, instructors_bei, subjects_bct, subjects_bei):
    timetable_matrix = [[""] + TIMESLOTS]  # Include timeslots in the first row

    for day in range(len(DAYS)):
        row = [DAYS[day]]
        for period in range(TOTAL_PERIODS):
            if period == 2 or period == 5:
                row.append("Break")
                continue

            instructor = int(timetable[period, day])
            if instructor >= 0 and instructor < len(instructors_bct):
                subject = subjects_bct[instructor]
                instructor_name = instructors_bct[instructor]
                row.append(f"{subject} ({instructor_name})")
            elif instructor >= len(instructors_bct) and instructor < len(instructors_bct) + len(instructors_bei):
                subject = subjects_bei[instructor - len(instructors_bct)]
                instructor_name = instructors_bei[instructor - len(instructors_bct)]
                row.append(f"{subject} ({instructor_name})")
            else:
                row.append("")

        timetable_matrix.append(row)

    return timetable_matrix

# Custom PDF class
class PDF(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 12)
        self.set_xy(0, 10)  # Set position at the top of the page
        self.cell(0, 10, 'Timetable', 0, 0, 'C')

    def chapter_title(self, title):
        self.set_font('Helvetica', 'B', 12)
        # Set position for the title, align it to the left
        self.set_xy(10, self.get_y())  # Position it slightly to the right
        self.cell(0, 10, title, 0, 1, 'L')  # Align left and move to next line
        self.ln(10)  # Add line break for spacing


    def timetable_table(self, timetable_matrix):
        self.set_font('Helvetica', '', 10)
        col_width = self.w / len(timetable_matrix[0]) - 1
        row_height = self.font_size

        for row in timetable_matrix:
            for cell in row:
                self.cell(col_width, row_height, cell, border=1)
            self.ln(row_height)

# Main function
def main():
    data = load_or_create_data()

    instructors_bct = data['instructors_bct']
    subjects_bct = data['subjects_bct']
    instructor_type_bct = data['instructor_type_bct']
    labs_bct = data['labs_bct']

    instructors_bei = data['instructors_bei']
    subjects_bei = data['subjects_bei']
    instructor_type_bei = data['instructor_type_bei']
    labs_bei = data['labs_bei']

    num_classes = len(DAYS)
    class_names = ["BCT", "BEI"]

    # Generate timetables
    timetable_bct = generate_timetable(num_classes, len(instructors_bct), len(instructors_bei), subjects_bct, subjects_bei, instructor_type_bct, instructor_type_bei, labs_bct, labs_bei)
    timetable_bei = generate_timetable(num_classes, len(instructors_bct), len(instructors_bei), subjects_bct, subjects_bei, instructor_type_bct, instructor_type_bei, labs_bct, labs_bei)

    # Display timetables
    timetable_matrix_bct = display_timetable(timetable_bct, num_classes, class_names, instructors_bct, instructors_bei, subjects_bct, subjects_bei)
    timetable_matrix_bei = display_timetable(timetable_bei, num_classes, class_names, instructors_bct, instructors_bei, subjects_bct, subjects_bei)

    # Create PDF
    pdf = PDF(orientation='L', format='A4')  # Landscape orientation
    pdf.add_page()
    pdf.chapter_title("BCT Timetable")
    pdf.timetable_table(timetable_matrix_bct)
    
    pdf.add_page()
    pdf.chapter_title("BEI Timetable")
    pdf.timetable_table(timetable_matrix_bei)

    pdf.output("timetable.pdf")

if __name__ == "__main__":
    main()

