

def print_loss(loss, iteration):
    print("Iteration : "+str(iteration)+" Loss : "+str(loss.item()))


def print_invalid_points_count(n, count):
    print("Out of "+str(n)+" points "+str(count)+" invalid points found.")


def print_total_number_of_points(n):
    print("In total "+str(n)+" points are in training data.")
