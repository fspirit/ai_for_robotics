from math import *
import numpy as np
import random
import copy

landmarks = [[0.0, 100.0], [0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]  # position of 4 landmarks in (y, x) format.
world_size = 100.0  # world is NOT cyclic. Robot is allowed to travel "out of bounds"

class CarParticle(object):

    def __init__(self, length=20.0, bearing_noise=0.1, steering_noise=0.1, distance_noise=1.0):

        self.x = 0.
        self.y = 0.
        self.orientation = pi / 4.0

        self.length = length  # length of a car

        self.bearing_noise = bearing_noise
        self.steering_noise = steering_noise
        self.distance_noise = distance_noise

    def set_random_state(self):
        self.x = random.random() * world_size  # initial x position
        self.y = random.random() * world_size  # initial y position
        self.orientation = random.random() * 2.0 * pi  # initial orientation

    def __repr__(self):
        return '[x=%.6s y=%.6s orient=%.6s]' % (str(self.x), str(self.y), str(self.orientation))

    def move(self, steering_angle, distance, add_noise=True):

        # add some noise

        if add_noise:
            steering_angle_n = random.gauss(steering_angle, self.steering_noise)
            distance_n = random.gauss(distance, self.distance_noise)
        else:
            steering_angle_n = steering_angle
            distance_n = distance

        # turning angle

        turn = tan(steering_angle_n) * distance_n / self.length

        if turn > 0.001:

            # the car is turning

            turning_radius = distance_n / turn
            cx = self.x - sin(self.orientation) * turning_radius
            cy = self.y + cos(self.orientation) * turning_radius
            self.orientation = (self.orientation + turn) % (2.0 * pi)
            self.x = cx + sin(self.orientation) * turning_radius
            self.y = cy - cos(self.orientation) * turning_radius

        else:

            # the car is going straight

            self.x += distance_n * cos(self.orientation)
            self.y += distance_n * sin(self.orientation)
            self.orientation = (self.orientation + turn) % (2.0 * pi)

        return self

    def estimate_bearing_angle(self, landmark, add_noise=True):

        bearing_angle = atan2(landmark[0] - self.y, landmark[1] - self.x) - self.orientation
        if add_noise:
            bearing_angle = random.gauss(bearing_angle, self.bearing_noise)
        bearing_angle %= (2.0 * pi)
        return bearing_angle

    def sense(self, add_noise=True):

        return [self.estimate_bearing_angle(landmark, add_noise) for landmark in landmarks]

    def measurement_prob(self, measurement):

        # calculate the correct measurement

        predicted_measurement = self.sense(add_noise=False)

        # compute errors

        error = 1.0
        for i in range(len(measurement)):
            error_bearing = abs(measurement[i] - predicted_measurement[i])
            error_bearing = (error_bearing + pi) % (2.0 * pi) - pi  # truncate

            # update Gaussian
            error *= exp(- (error_bearing ** 2) / (self.bearing_noise ** 2) / 2.0) / sqrt(2.0 * pi * (self.bearing_noise ** 2))

        return error

    def get_position(self):
        return self.x, self.y, self.orientation


class ParticleFilter2D(object):

    def __init__(self, particles_num=500):
        self.particles = [CarParticle() for _ in range(particles_num)]
        for p in self.particles:
            p.set_random_state()

    def update(self, motion, measurement):

        # move particles

        self.particles = [p.move(*motion) for p in self.particles]

        # resample particles

        self.resample(measurement)

    def resample(self, measurement):

        # get particles likelihoods

        w = [p.measurement_prob(measurement) for p in self.particles]

        # resample particles according to their likelihood

        index = random.randint(0, len(self.particles) - 1)
        beta = 0.
        resampled_particles = []
        max_w = max(w)
        for _ in self.particles:
            beta += random.random() * 2.0 * max_w
            while w[index] < beta:
                beta -= w[index]
                index = (index + 1) % len(self.particles)
            resampled_particles.append(copy.copy(self.particles[index]))

        self.particles = resampled_particles

    def get_position(self):
        anchor = self.particles[0].orientation
        return np.mean([(p.x, p.y, (((p.orientation - anchor + pi) % (2.0 * pi)) + anchor - pi)) for p in self.particles],
                       axis=0)

if __name__ == "__main__":

    number_of_iterations = 8
    motions = [[2. * pi / 10, 20.] for row in range(number_of_iterations)]

    # generate ground truth

    the_car = CarParticle()
    the_car.set_random_state()
    ground_truth = [the_car.move(*m).sense() for m in motions]

    final_gt_position = the_car.get_position()

    # run particle filter

    filter = ParticleFilter2D()
    for motion, measurement in zip(motions, ground_truth):
        filter.update(motion, measurement)

    final_estimated_position = filter.get_position()

    # print results

    print("Ground truth:")
    print(final_gt_position)
    print("Estimated position:")
    print(final_estimated_position)

    est_error = (abs(final_gt_position[0] - final_estimated_position[0]),
                abs(final_gt_position[1] - final_estimated_position[1]),
                (final_gt_position[2] - final_estimated_position[2] + pi) % (2.0 * pi) - pi)

    print("Error:")
    print(est_error)
