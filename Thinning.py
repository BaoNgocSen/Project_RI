import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
import math
class thinning_hawkes(object):
    """
    Univariate Hawkes process with different kernels.
    No events or initial condition before initial time.
    """

    def __init__(self, mu, alpha, beta, T, last_jumps=None, Process=None, kernel_type='exponential', peri=2.0,epsi=0.3):
        self.mu = mu
        self.alpha = alpha
        self.beta = beta
        self.T = T
        self.lambda_max = mu
        self.simulated = False
        self.kernel_type = kernel_type
        self.Process = []
        self.peri = peri
        self.epsilon=epsi

    def simulate(self):
        if self.simulated:
            print("We've already simulated this object")
            return self.Process

        self.Process = []
        self.simulated = True
        current_t = 0.0
        I_decayed_sum = 0.0

        while current_t < self.T:
            if self.kernel_type == 'exponential':
                #lambda(t) = mu + sum_{t_k < t} alpha * exp( -beta * (t - t_k) )
                upper_intensity = self.mu + I_decayed_sum
                u1 = np.random.uniform(0, 1)
                dt = -math.log(u1) / upper_intensity
                t_candidate = current_t + dt
                if t_candidate >= self.T:
                    break

                I_decayed_sum *= math.exp(-self.beta * (t_candidate - current_t))
                lam_at_t_candidate = self.mu + I_decayed_sum
                u2 = np.random.uniform(0, 1)

                if u2 <= lam_at_t_candidate / upper_intensity:
                    self.Process.append(t_candidate)
                    I_decayed_sum += self.alpha
                    current_t = t_candidate
                else:
                    current_t = t_candidate

            elif self.kernel_type == 'exponential_periodic':
                """
                Let P = peri. At every t = m*P, the history is cleared (reset).
                lambda(t) = mu + sum_{kP <= t_k < t} alpha * exp( -beta * (t - t_k) )              
                """

                # Next reset intensity moment
                k_current = math.floor(current_t / self.peri)
                next_reset_time = (k_current + 1) * self.peri
    
                upper_intensity = self.mu + I_decayed_sum
                u1 = np.random.uniform(0, 1)
                dt = -math.log(u1) / upper_intensity
                t_candidate = current_t + dt

                if t_candidate >= self.T:
                    break

                # Check if candidate exceeds reset time, if it reach reset point: clear history and restart
                if t_candidate >= next_reset_time:
                    current_t = next_reset_time
                    I_decayed_sum = 0.0
                    continue
                    
                #Else thinning as usual
                I_decayed_sum *= math.exp(-self.beta * (t_candidate - current_t))
                lam_at_t_candidate = self.mu + I_decayed_sum
                u2 = np.random.uniform(0, 1)

                if u2 <= (lam_at_t_candidate / upper_intensity):
                    self.Process.append(t_candidate)
                    I_decayed_sum += self.alpha
                
                current_t = t_candidate

            elif self.kernel_type == 'gaussien':
                #lambda(t) = mu + sum_{t_k < t} alpha * exp( -(t - t_k)^2 / (2 * beta^2) )
          
                lam_current = self.mu + sum(
                    self.alpha * np.exp(-((current_t - tk) ** 2) / (2 * self.beta ** 2)) for tk in self.Process)
                
                upper_intensity = lam_current + self.alpha
                
                if upper_intensity <= 0:
                    current_t = self.T
                    break

                u = np.random.uniform(0, 1)
                w = -np.log(u) / upper_intensity
                t_candidate = current_t + w
                if t_candidate >= self.T:
                    break

                lam_candidate = self.mu + sum(
                    self.alpha * np.exp(-((t_candidate - tk) ** 2) / (2 * self.beta ** 2)) for tk in self.Process  )

                D = np.random.uniform(0, 1)
                if D <= lam_candidate / upper_intensity:
                    self.Process.append(t_candidate)
                    current_t = t_candidate
                else:
                    current_t = t_candidate

            elif self.kernel_type == 'box_periodic':
                """   
                Let P = peri.At every t = m*P, history is reset.
                lambda(t) = mu + alpha * #{k : kP <= t_k < t  and  t - t_k < beta}

                """
                # Determine current period and next reset time 
                k_current = math.floor(current_t / self.peri)
                start_of_period = k_current * self.peri
                next_reset_time = (k_current + 1) * self.peri
                
                # Count active events in current period
                num_active_events = 0
                cutoff = max(start_of_period, current_t - self.beta)
        
                
                for tk in reversed(self.Process):
                    if tk > cutoff:
                        num_active_events += 1
                    else:
                        break

                upper_intensity = self.mu + self.alpha * num_active_events
                
                if upper_intensity <= 0:
                    current_t = next_reset_time
                    continue

                w = -np.log(np.random.random()) / upper_intensity
                t_candidate = current_t + w

                if t_candidate >= self.T:
                    break

                if t_candidate >= next_reset_time:
                    current_t = next_reset_time
                    continue

                num_active_at_cand = 0
                cutoff_cand = max(start_of_period, t_candidate - self.beta)
                
                for tk in reversed(self.Process):
                    if tk > cutoff_cand:
                        num_active_at_cand += 1
                    else:
                        break

                true_intensity = self.mu + self.alpha * num_active_at_cand
                
                if np.random.random() <= (true_intensity / upper_intensity):
                    self.Process.append(t_candidate)
                
                current_t = t_candidate   

            elif self.kernel_type == 'inhibitory_exponential':
                   #lambda(t) = max( epsilon , mu - sum_{t_k < t} alpha * exp( -beta * (t - t_k) ) ) 
                    
                upper_intensity = max(self.mu,self.epsilon)
                
                u1 = np.random.uniform(0, 1)
                dt = -math.log(u1) / max(upper_intensity, 1e-9)
                t_candidate = current_t + dt
                
                if t_candidate >= self.T: 
                    break

                I_decayed_sum *= math.exp(-self.beta * (t_candidate - current_t))

                lam_at_t_candidate = max(self.epsilon, self.mu - I_decayed_sum)
                
                if np.random.uniform(0, 1) <= lam_at_t_candidate / upper_intensity:
                    self.Process.append(t_candidate)
                    I_decayed_sum += self.alpha
                current_t = t_candidate
                    
            elif self.kernel_type == 'box_decale':
                #lambda(t) = mu + alpha * #{ k : epsilon < t - t_k < epsilon + beta }
                mu, alpha, beta, T,epsilon = self.mu, self.alpha, self.beta, self.T,self.epsilon
                num_active_events = 0
                cut_avant = current_t - beta-epsilon
                cutoff= current_t- epsilon   
                for tk in reversed(self.Process):
                    if tk < cutoff and tk > cut_avant:
                        num_active_events += 1
                    else:
                        break

                upper_intensity = mu + alpha * num_active_events
                if upper_intensity <= 0:
                    current_t = T
                    break

                w = -np.log(np.random.random()) / upper_intensity
                t_candidate = current_t + w
                if t_candidate >= T:
                    break

                num_active_at_cand = 0
                cutoff_cand = t_candidate - epsilon  
                cut_avant_cand= t_candidate - beta-epsilon
                for tk in reversed(self.Process):
                    if tk < cutoff_cand and tk > cut_avant_cand :
                        num_active_at_cand += 1
                    else:
                        break

                true_intensity = mu + alpha * num_active_at_cand
                if np.random.random() <= (true_intensity / upper_intensity):
                    self.Process.append(t_candidate)  
                current_t = t_candidate     
        return self.Process

    def intensity_at_t(self, t):
        if not self.simulated:
            raise RuntimeError("Simulate first before calling intensity_at_t()")
        A = self.mu
        
        if self.kernel_type == 'exponential':
            for k in self.Process:
                if k < t:
                    A += self.alpha * np.exp(-self.beta * (t - k))
                else: break
        
        elif self.kernel_type == 'exponential_periodic':
            k_period = math.floor(t / self.peri)
            start_of_period = k_period * self.peri
            for k in self.Process:
                if start_of_period <= k < t:
                    A += self.alpha * np.exp(-self.beta * (t - k))
                elif k >= t: break

        elif self.kernel_type == 'gaussien':
            for k in self.Process:
                if k < t:
                    A += self.alpha * np.exp(-((t - k) ** 2) / (2 * self.beta ** 2))
                else: break

        elif self.kernel_type == 'box_periodic':
            k_period = math.floor(t / self.peri)
            start_of_period = k_period * self.peri
            lower_bound = max(start_of_period, t - self.beta)
            
            count = 0
            for k in reversed(self.Process):
                if k < lower_bound:
                    break
                if k < t:
                    count += 1
            
            A += self.alpha * count
                
        elif self.kernel_type == 'inhibitory_exponential':
            total_inhibition = 0
            for tk in reversed(self.Process):
                if tk < t:
                    influence = self.alpha * np.exp(-self.beta * (t - tk))
                    if influence < 1e-7: break 
                    total_inhibition += influence
        
            return max(self.epsilon, self.mu - total_inhibition)
        elif self.kernel_type == 'box_decale':
            A += self.alpha * sum(1 for k in self.Process if self.epsilon < t - k < self.beta+ self.epsilon)
        else:
            raise ValueError(f"Unknown kernel_type={self.kernel_type}")
            
        return A

    def plot_intensity(self, steps, t):
        if t > self.T:
            print("t is greater than T")
            return
        if not self.simulated:
            print("Simulate first")
            return

        x = np.linspace(0, t, steps)
        y = [self.intensity_at_t(i) for i in x]

        plt.figure(figsize=(8, 4))
        plt.plot(x, y, lw=1.5)
        plt.xlabel("Time")
        plt.ylabel("Intensity λ(t)")
        plt.title(f"Hawkes intensity — kernel={self.kernel_type}, T={self.T}, mu={self.mu}, alpha={self.alpha}, beta={self.beta}") 
        plt.tight_layout()
        plt.show()