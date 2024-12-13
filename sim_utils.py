def read_processed_input(f: str):
    fin = open(f)
    if fin.closed:
        print("Bad input filename given for read_processed_input:" + f)
        return

    sol = []
    for _ in range(3):
        sol.append(list(map(int, fin.readline().split())))

    fin.close()
    return sol #[split, push_up, push_down]

#scrie date procesate intr-un fisier astfel incat sa se poata citi de tminterface
def write_processed_output(fName: str, sol, GAP_TIME):
    fout = open(fName, "w")
    if fout.closed:
        print("Couldn't create file with name: " + fName)
        return

    steer, push_up, push_down = sol
    assert(len(steer) == len(push_up) and len(push_up) == len(push_down))
    n = len(steer)

    for push, dir in [(push_up, "up"), (push_down, "down")]:
        i = 0
        while i < n:
            while i < n and push[i] == 0:
                i += 1
            if i < n:
                j = i
                while j < n and push[j] == 1:
                    j += 1
                fout.write(str(i * GAP_TIME) + "-" + str(j * GAP_TIME) + " press " + dir + "\n")
                i = j

    for i in range(n):
        fout.write(str(i * GAP_TIME) + " steer " + str(steer[i]) + "\n")

    fout.close()
