# with 4 q_ul

# Section 1: Import from modules and define a utility class.

from collections import deque # double-ended queue
import numpy as np
import simpy
from simpy.util import start_delayed
import random
from display import printRoad
from datetime import time, timedelta, datetime
import pandas as pd

class Struct(object):
   """
   This simple class allows one to create an object whose attributes are
   initialized via keyword argument/value pairs.  One can update the attributes
   as needed later.
   """
   def __init__(self, **kwargs):
      self.__dict__.update(kwargs)

# Section 2: Initializations.

np.random.seed(55)

# Total number of seconds to be simulated:
end_time= 24*60*60 # one day

# Cars cars arrive at the traffic light according to a Poisson process with an
# average rate of 0.2 per second:
arrival_rate= 0.2
t_interarrival_mean= 1.0 / arrival_rate

# Traffic light green and red durations:
t_green= 30.0; t_red= 40.0

# The time for a car at the head of the queue to depart (clear the intersection)
# is modeled as a triangular distribution with specified minimum, maximum, and
# mode.
t_depart_left= 1.6; t_depart_mode= 2.0; t_depart_right= 2.4

manual = False



# Track number of cars:
arrival_count= departure_count= 0

Q_stats= Struct(count=0, cars_waiting=0)
Q_stats_N = Struct(count=0, cars_waiting=0)
Q_stats_W = Struct(count=0, cars_waiting=0)
Q_stats_S = Struct(count=0, cars_waiting=0)
Q_stats_E = Struct(count=0, cars_waiting=0)
W_stats= Struct(count=0, waiting_time=0.0)
W_stats_min= Struct(count=0, waiting_times =[],waiting_time=0.0)
notes = []

class intersection(object):
    "green light = True, red light = False"
    intersection_north = None
    intersection_south = None
    intersection_west = None
    intersection_east = None

    @property
    def light_horizontal(self):
        return not self.light_vertical

    def __repr__(self):
        return f"@{self.name} ver = {self.light_vertical} hor = {self.light_horizontal}"

    def __init__(self, env: simpy.Environment, name):
        self.name = name
        self.light_vertical = True
        self.queue_W = deque()
        self.queue_N = deque()
        self.queue_E = deque()
        self.queue_S = deque()
        self.env = env

        if not manual:
            self.action = env.process(self.run())

        # Schedule first arrival of a car:
        t_first_arrival = np.random.exponential(t_interarrival_mean)
        start_delayed(env, self.arrival(), delay=t_first_arrival)

    def switch_light(self):
        self.light_vertical = not self.light_vertical
        # if self.light_vertical:
        #     print("\nThe light turned green vertically at time %.3f." % self.env.now)
        # else:
        #     print("\nThe light turned green horizontally at time %.3f." % self.env.now)

    # Section 4.2: Light change-of-state event.

    def departure(self, queue):
        """
        This generator function simulates the 'departure' of a car, i.e., a car that
        previously entered the intersection clears the intersection.  Once a car has
        departed, we remove it from the queue, and we no longer track it in the
        simulation.
        """
        env = self.env
        if len(queue) == 0:
            return

        while True:

            # The car that entered the intersection clears the intersection:
            if len(queue) == 0:
                return
            car_number, t_arrival = queue.popleft()
            # print("Car #%d departed at time %.3f, leaving %d cars in the queue."
            #       % (car_number, env.now, len(queue)))

            # Record waiting time statistics:
            W_stats.count += 1
            W_stats.waiting_time += env.now - t_arrival

            if W_stats_min.count < 60:
                W_stats_min.count += 1
                W_stats_min.waiting_times.append(env.now - t_arrival)
                W_stats_min.waiting_time = np.mean(W_stats_min.waiting_times)
            else:
                W_stats_min.waiting_times.append(env.now - t_arrival)
                W_stats_min.waiting_times = W_stats_min.waiting_times[1:]
                W_stats_min.waiting_time = np.mean(W_stats_min.waiting_times)


            # If the light is red or the queue is empty, do not schedule the next
            # departure.  `departure` is a generator, so the `return` statement
            # terminates the iterator that the generator produces.
            if queue is self.queue_S or queue is self.queue_N:
                light = self.light_horizontal
            else:
                light = self.light_vertical

            if not light or len(queue) == 0:
                return

            # Generate departure delay as a random draw from triangular distribution:
            delay = np.random.triangular(left=t_depart_left, mode=t_depart_mode,
                                      right=t_depart_right)

            # Schedule next departure:
            yield env.timeout(delay)

    def arrival(self, direction=None):
        """
        This generator functions simulates the arrival of a car.  Cars arrive
        according to a Poisson process having rate `arrival_rate`.  The times between
        subsequent arrivals are i.i.d. exponential random variables with mean

          t_interarrival_mean= 1.0 / arrival_rate
        """
        global arrival_count
        env = self.env

        while True:
            arrival_count += 1

            direction = random.choice(['W', 'E', 'N', 'S'])

            if direction in ['N', 'W']:
                light = self.light_horizontal
            else:
                light = self.light_vertical
            queue = eval(f'self.queue_{direction}')

            if not light or len(queue):

                # The light is red or there is a queue of cars.  ==> The new car joins
                # the queue.  Append a tuple that contains the number of the car and
                # the time at which it arrived:
                queue.append((arrival_count, env.now))
                # print("Car #%d arrived and joined the queue at position %d at time "
                #       "%.3f." % (arrival_count, len(queue), env.now))

            else:

                # The light is green and no cars are waiting.  ==> The new car passes
                # through the intersection immediately.
                # print("Car #%d arrived to a green light with no cars waiting at time "
                #       "%.3f." % (arrival_count, env.now))

                # Record waiting time statistics.  (This car experienced zero waiting
                # time, so we increment the count of cars, but the cumulative waiting
                # time remains unchanged.
                W_stats.count += 1

                if W_stats_min.count < 60:
                    W_stats_min.count += 1
                    W_stats_min.waiting_times.append(0)
                    W_stats_min.waiting_time = np.mean(W_stats_min.waiting_times)
                else:
                    W_stats_min.waiting_times.append(0)
                    W_stats_min.waiting_times = W_stats_min.waiting_times[1:]
                    W_stats_min.waiting_time = np.mean(W_stats_min.waiting_times)

            # Schedule next arrival:
            interval = np.random.exponential(t_interarrival_mean)
            yield env.timeout(interval)

    def run(self):
        """
        This generator function simulates state changes of the traffic light.  For
        simplicity, the light is either green or red--there is no yellow state.
        """


        while True:

            # Section 4.2.1: Change the light to green.

            self.switch_light()

            # If there are cars in the queue, schedule a departure event:
            if self.light_vertical:
                if len(self.queue_W):
                    # Generate departure delay as a random draw from triangular
                    # distribution:
                    delay = np.random.triangular(left=t_depart_left, mode=t_depart_mode,
                                              right=t_depart_right)

                    start_delayed(self.env, self.departure(self.queue_W), delay=delay)
                if len(self.queue_E):
                    # Generate departure delay as a random draw from triangular
                    # distribution:
                    delay = np.random.triangular(left=t_depart_left, mode=t_depart_mode,
                                              right=t_depart_right)

                    start_delayed(self.env, self.departure(self.queue_E), delay=delay)
            else:
                if len(self.queue_N):
                    # Generate departure delay as a random draw from triangular
                    # distribution:
                    delay = np.random.triangular(left=t_depart_left, mode=t_depart_mode,
                                              right=t_depart_right)

                    start_delayed(self.env, self.departure(self.queue_N), delay=delay)

                if len(self.queue_S):
                    # Generate departure delay as a random draw from triangular
                    # distribution:
                    delay = np.random.triangular(left=t_depart_left, mode=t_depart_mode,
                                              right=t_depart_right)

                    start_delayed(self.env, self.departure(self.queue_S), delay=delay)



            # Schedule event that will turn the light red:
            if self.light_vertical:
                yield self.env.timeout(t_green)
            else:
                yield self.env.timeout(t_red)

    def switchAndDepart(self):
        """
        This generator function simulates state changes of the traffic light.  For
        simplicity, the light is either green or red--there is no yellow state.
        """

        # Section 4.2.1: Change the light to green.

        self.switch_light()

        # If there are cars in the queue, schedule a departure event:
        if self.light_vertical:
            if len(self.queue_W):
                # Generate departure delay as a random draw from triangular
                # distribution:
                delay = np.random.triangular(left=t_depart_left, mode=t_depart_mode,
                                          right=t_depart_right)

                start_delayed(self.env, self.departure(self.queue_W), delay=delay)
            if len(self.queue_E):
                # Generate departure delay as a random draw from triangular
                # distribution:
                delay = np.random.triangular(left=t_depart_left, mode=t_depart_mode,
                                          right=t_depart_right)

                start_delayed(self.env, self.departure(self.queue_E), delay=delay)
        else:
            if len(self.queue_N):
                # Generate departure delay as a random draw from triangular
                # distribution:
                delay = np.random.triangular(left=t_depart_left, mode=t_depart_mode,
                                          right=t_depart_right)

                start_delayed(self.env, self.departure(self.queue_N), delay=delay)

            if len(self.queue_S):
                # Generate departure delay as a random draw from triangular
                # distribution:
                delay = np.random.triangular(left=t_depart_left, mode=t_depart_mode,
                                          right=t_depart_right)

                start_delayed(self.env, self.departure(self.queue_S), delay=delay)



def monitor(intersec: intersection):
   """
   This generator function produces an interator that collects statistics on the
   state of the queue at regular intervals.  An alternative approach would be to
   apply the PASTA property of the Poisson process ('Poisson Arrivals See Time
   Averages') and sample the queue at instants immediately prior to arrivals.
   """
   global env, Q_stats, notes

   notes = []

   while True:
      Q_stats.count+= 1
      Q_stats.cars_waiting+= (len(intersec.queue_N) + len(intersec.queue_S) + len(intersec.queue_W) + len(intersec.queue_E))/4
      Q_stats_E.cars_waiting+=len(intersec.queue_E)
      Q_stats_N.cars_waiting+=len(intersec.queue_N)
      Q_stats_S.cars_waiting+=len(intersec.queue_S)
      Q_stats_W.cars_waiting+=len(intersec.queue_W)
      notes.append({
          "waiting_time_min": W_stats_min.waiting_time,
          "sec": env.now,
          "time": time_plus(time(), timedelta(seconds=env.now))
      })

      yield env.timeout(1.0)

def change_arrival_rate(peak=1.0, base=0.3):
    global arrival_rate, t_interarrival_mean, env
    topping = peak - base
    while True:
        # Cars cars arrive at the traffic light according to a Poisson process with an
        # average rate of 0.2 per second:
        if env.now < 8*60*60:
            arrival_rate = base + env.now/(8*60*60) * topping
        elif env.now < 8*60*60 + 4.5*60*60:
            arrival_rate = base + (env.now-8 * 60 * 60) / (4.5*60*60) * topping
        elif env.now < 17*60*60:
            arrival_rate = base + (env.now-(8*60*60 + 4.5*60*60)) / (17 * 60 * 60 - (8*60*60 + 4.5*60*60)) * topping
        else:
            arrival_rate = base + (env.now - 17 * 60 * 60) / (7 * 60 * 60) * topping

        assert arrival_rate < peak + 0.1

        t_interarrival_mean = 1.0 / arrival_rate
        yield env.timeout(1.0)

def test_intersection():
    a = intersection()
    print(a)
    a.switch_light()
    print(a)
    a.switch_light()
    print(a)

def run():
    # Section 5: Schedule initial events and run the simulation.  Note: The first
    # change of the traffic light, first arrival of a car, and first statistical
    # monitoring event are scheduled by invoking `env.process`.  Subsequent changes
    # will be scheduled by invoking the `timeout` method.  With this scheme, there
    # is only one event of each of these types scheduled at any time; this keeps the
    # event queue short, which is good for both memory utilization and running time.

    global env
    print("\nSimulation of Cars Arriving at Intersection Controlled by a Traffic "
          "Light\n\n")

    # Initialize environment:
    env = simpy.Environment()

    intersecA = intersection(env,'intersec A')
    # Schedule first statistical monitoring event:
    env.process(monitor(intersecA))
    env.process(change_arrival_rate())
    # Let the simulation run for specified time:
    env.run(until=end_time)

    # Section 6: Report statistics.

    print("\n\n      *** Statistics ***\n\n")

    print("Mean number of cars waiting: %.3f"
          % (Q_stats.cars_waiting / float(Q_stats.count)))

    print("Mean waiting time (seconds): %.3f"
          % (W_stats.waiting_time / float(W_stats.count)))

def init(manual=False):
    global env, intersecA
    print("\nSimulation of Cars Arriving at Intersection Controlled by a Traffic "
          "Light\n\n")

    # Initialize environment:
    env = simpy.Environment()

    intersecA = intersection(env, 'intersec A')
    # Schedule first statistical monitoring event:
    env.process(monitor(intersecA))
    env.process(change_arrival_rate())

def time_plus(time, timedelta):
    start = datetime(
        2000, 1, 1,
        hour=time.hour, minute=time.minute, second=time.second)
    end = start + timedelta
    return end.time()

def getState():
    global env,intersecA
    t = time_plus(time(), timedelta(seconds=env.now))
    return len(intersecA.queue_N), len(intersecA.queue_E), len(intersecA.queue_W), len(intersecA.queue_S), str(t)

def actionStepAndGetState(nparray):
    global env,intersecA

    willSwitch = nparray[0]
    stayStill = nparray[1]
    if willSwitch > stayStill:
        intersecA.switchAndDepart()
    else:
        pass
    env.step()
    t = time_plus(time(), timedelta(seconds=env.now))
    reward = W_stats_min.waiting_time * -1 # mean of car
    return len(intersecA.queue_N), len(intersecA.queue_E), len(intersecA.queue_W), len(intersecA.queue_S), env.now, reward, env.now > end_time

def action():
    pass

def print_stats():
    # Section 6: Report statistics.

    # write csv
    df = pd.DataFrame(notes)
    df.to_csv('waiting_time_min.csv',index=False)
    print("\n\n      *** Statistics ***\n\n")

    print("Mean number of cars waiting: %.3f"
          % (Q_stats.cars_waiting / float(Q_stats.count)))

    print("Mean waiting time (seconds): %.3f"
          % (W_stats.waiting_time / float(W_stats.count)))

if __name__ == '__main__':
    from time import sleep
    init()
    while env.now <= end_time:
        #sleep(0.001)
        env.step()
        if env.now % 60 == 0:
            printRoad(*getState())

    print_stats()