class Student:
    def __init__(self, name, grade, percentage):
        self.name=name
        self.grade = grade
        self.__percentage = percentage
    
    def get_percentage(self):
        return self.__percentage
    

    def student_details(self):
        print(f"{self.name} is in class {self.grade} and has scored {self.get_percentage()}% marks")

    


# student1 = Student('john',12,84)
# student2 = Student('peter',11,56)
# student1.student_details()
# student2.student_details()

# print(student1.__dict__)

class graduateStudent(Student):
    def __init__(self,name,age,grade,percentage,stream):
        super().__init__(name,grade,percentage)
        self.age= age
        self.stream = stream
    
    def student_details(self):
        super().student_details()
        print(f"He is {self.age} years old and has choosen {self.stream} ")


Grad_student1 = graduateStudent('john',21,12,99,'science')

Grad_student1.student_details()

print(Grad_student1.get_percentage())

