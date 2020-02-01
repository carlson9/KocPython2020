# - Add a student's name to the roster for a grade
# - Get a list of all students enrolled in a grade
# - Get a sorted list of all students in all grades.
# 
# Note that all our students only have one name.
# (It's a small town, what do you want?)

class school():
    def __init__(self, school_name): #initialize instance of class School with parameter name
        self.school_name = school_name #user must put name, no default
        self.db = {} #initialize empty dictionary to store kids and grades
        
    def add(self, name, student_grade): #add a kid to a grade in instance of School
        if student_grade in self.db: #need to check if the key for the grade already exists, otherwise assigning it will return error
            self.db[student_grade].add(name) #add kid to the set of kids within the dictionary
        else: self.db[student_grade] = {name} #if the key doesn't exist, create it and put kid in

    def sort(self): #sorts kids alphabetically and returns them in tuples (because they are immutable)
        sorted_students={} #sets up empty dictionary to store sorted tuples
        for key in self.db.keys(): #loop through each key
            sorted_students[key] = tuple(sorted(self.db[key])) #add dictionary entry with key being the grade and the entry the tuple of kids
        return sorted_students

    def grade(self, check_grade):
        if check_grade not in self.db: return None #if the key doesn't exist, there are no kids in that grade: return None
        return self.db[check_grade] #if None wasn't returned above, return elements within dictionary, or kids in grade

    def __str__(self): #print function will display the school name on one line, and sorted kids on other line
        return "%s\n%s" %(self.school_name, self.sort())
