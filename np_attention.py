import math
import torch
from torch import nn
import numpy as np
import random

torch.manual_seed(0)
np.random.seed(0)
nprng = np.random.default_rng()
random.seed(0)
try:
    from arsenal.datastructures.heap.sumheap import SumHeap
except:
    print("WARNING: package arsenal is not installed, can only run with fast_sampling = False")

class attention_kernel(nn.Module):
    def __init__(self,
        D:int,
        S:float,
        P:int,
        L:int,
        H:int,
        method: str,    
        fast_sampling = True,
        verbose: bool = False,                                           
        **kwargs,
    ):
        """

        Attributes:
            Parameters:    
            D (int): Projection dimension
            S (float): Attention inner-product scaling
            P (int): Parallelisation (2^k for some k)
            L (int): Maximum sequence length (2^k for some k)
            H (int): Number of heads to process concurrently (better parallelism)
            fast_sampling (Optional[int]): whether to use a fast heap-based sampling intead of numpy.choice. Default is True.
            
            method (str): Specify method to get probability mass. One of "alignment", "performer", "averaged", "uniform"
            R (Optional[int]): Dimension of feature transformation for Performer

        Inplace Updates:
            k is added to K from leaf l to root
            v is added to V from leaf l to root
            num_sub_tree_leaves is incremented from leaf l to root
            (Optional) phi(k) is added to phi_K from leaf to root

        """    
        super().__init__()
        assert D > 0 and int(D) == D, "D must be a positive integer"
        assert S > 0, "score scaling of inner-products must be positive"
        assert P > 0 and math.log2(P).is_integer, "P must be a power of 2"
        assert L > 0 and math.log2(L).is_integer, "maximal length must be a power of 2"
        
        assert H > 0 and int(H) == H, "H must be a positive integer"

        self.D = D
        self.S = S
        self.P = P
        self.L = L
        self.H = H

        # l (int): Current Zero-indexed token position
        self.l = 0
        # K (torch.Tensor): Memory for perfect binary key-tree with L leaves, in \mathbb{R}^{H x 2L-1 x D}
        #self.register_buffer(name="K", tensor=torch.zeros(self.H, 2*self.L-1, self.D), persistent=False)
        self.K = np.zeros((H, 2*self.L-1, self.D))
        # V (torch.Tensor): Memory for perfect binary value-tree with L leaves, in \mathbb{R}^{H x 2L-1 x D}
        #self.register_buffer(name="V", tensor=torch.zeros(self.H, 2*self.L-1, self.D), persistent=False
        self.V = np.zeros((H, 2*self.L-1, self.D))
        # num_sub_tree_leaves (torch.Tensor): Tensor indicating how many leaves the subtree of a node contains, in \mathbb{N}^{1 x 2L-1}
        #self.register_buffer(name="num_sub_tree_leaves", tensor=torch.zeros(1, 2*self.L-1, dtype=np.int32), persistent=False)
        self.num_sub_tree_leaves = np.zeros((1, 2*self.L-1), dtype=int)

        self.fast_sampling = fast_sampling

        self.method = method

        # smallest nonzero number
        self.epsilon = 1e-10

        if self.method == "FAVOR+ReLU" or self.method == "FAVOR+":
            assert 'R' in kwargs, "when using inverse kernels, must specify additional parameter R"
            self.R = kwargs['R']
            # Phi_K: (Optional[(torch.Tensor)]) Tensor containing transformed keys when using kernel method, in \mathbb{R}^{H x 2L-1 x R}
            self.phi_K = np.zeros((H, 2*self.L-1, self.R))
            self.buds_phi_alignments = np.zeros((H, self.L))

        if self.method == "RandomFourierFeatures":
            assert 'R' in kwargs, "when using inverse kernels, must specify additional parameter R"
            self.R = kwargs['R']
            # Phi_K: (Optional[(torch.Tensor)]) Tensor containing transformed keys when using kernel method, in \mathbb{R}^{H x 2L-1 x R}
            self.phi_K = np.zeros((H, 2*self.L-1, 2*self.R))
            self.buds_phi_alignments = np.zeros((H, self.L))

        if self.method == "positive_alignment":
            self.R = 2*self.D # ReLU of positive and negative of key/query
            # Phi_K: (Optional[(torch.Tensor)]) Tensor containing transformed keys when using kernel method, in \mathbb{R}^{H x 2L-1 x R}
            self.phi_K = np.zeros((H, 2*self.L-1, self.R))
            self.buds_phi_alignments = np.zeros((H, self.L))
        
        if self.method == "exponentially_decaying_horizon":
            assert 'base' in kwargs, "when using time decay, must specify additional parameter base"
            self.base = kwargs['base']
            # of subtree at some index
            self.leftest_leaf_positions = -np.ones((2*self.L-1), dtype=int) # -1 encodes unspecified leftermost position
            self.rightest_leaf_positions = -np.ones((2*self.L-1), dtype=int)

        # these are overwritten during each forward call   
        self.buds_alignments = np.zeros((H, self.L))
        self.buds_indices = np.zeros(self.L, dtype=int)
        self.buds_pms = np.zeros(self.L)

        self.verbose = verbose
        self.print_last_tree = False

    #@profile
    def forward(self,
        T: int,
        k: torch.Tensor,
        q: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Main forward function for the algorithm, which runs using the current key, query and value.

        Args:
            T (int): Number of terms in approximation
            k (torch.Tensor): current key (at position l)
            q (torch.Tensor): current query (at position l)
            v (torch.Tensor): current value (at position l)

        Returns:
            torch.Tensor: the approximation of \sum_{j=0}^{l-1} v_j (\exp(<q_l, k_j>))/(\sum_{i=0}^{l-1}\exp(<q_l, k_i>)) for each head, in R^{H x D}
        """

        assert k.shape == (self.H, 1, self.D,) or k.shape == (self.H, self.D), "Current key must be of shape (H, D) or (H, 1, D)"
        assert q.shape == (self.H, 1, self.D,) or q.shape == (self.H, self.D), "Current query must be of shape (H, D) or (H, 1, D)"
        assert v.shape == (self.H, 1, self.D,) or v.shape == (self.H, self.D), "Current value must be of shape (H, D) or (H, 1, D)"
        assert int(T) == T and T >= 0 and T <= self.l, "number of terms in approximation must be an integer in [0, l]"

        k = torch.Tensor.numpy(k, force=True)
        q = torch.Tensor.numpy(q, force=True)
        v = torch.Tensor.numpy(v, force=True)
        
        k = np.reshape(k, newshape=(self.H, 1, self.D))
        q = np.reshape(q, newshape=(self.H, 1, self.D))
        v = np.reshape(v, newshape=(self.H, 1, self.D))

        Pred = 0

        if self.verbose == True and self.l == self.L-1:
            self.print_last_tree = True

        # method dependent pre_processing
        self.kernel_setup(k=k, q=q)

        if self.print_last_tree:
            print("==============================================================")
            print("New Run")
            print("l is ", self.l)

        if self.l != 0:
            # ensures that start_index is not beneath tree due to high P and low l
            if math.ceil(math.log2(self.l)) < math.log2(self.P):
                self.start_P = 2**(math.ceil(math.log2(self.l)))
            else:
                self.start_P = self.P
            start_index = int(2**(math.log2(self.L) - math.ceil(math.log2(self.l)) + math.log2(self.start_P))-1) # \sum_{k=0}^{\log_2 L - \lceil \log_2 l \rceil + \log_2 P - 1} 2^k
            # ensures that we do not start in a masked part of the tree
            self.start_P = np.count_nonzero(self.num_sub_tree_leaves[0, start_index:start_index+self.start_P])
            assert self.start_P <= self.P, "start P should always be smaller than self.P"

            if self.print_last_tree:
                print("Starting at start index: ", start_index)

            self.buds_alignments[:, :self.start_P] = np.sum(q * self.K[:, start_index:start_index + self.start_P, :], axis=-1)
            if self.method == "FAVOR+ReLU" or self.method == "FAVOR+" or self.method == "RandomFourierFeatures" or self.method == "positive_alignment":
                self.buds_phi_alignments[:, :self.start_P] = np.sum(self.phi_q * self.phi_K[:, start_index:start_index + self.start_P, :], axis=-1)

            initial_buds_indices = np.arange(start=start_index, stop=start_index + self.start_P, dtype=int)

            self.buds_indices[:self.start_P] = initial_buds_indices
            self.buds_pms[:self.start_P] = self.get_pm(alignments=self.buds_alignments[:, :self.start_P] if self.method == "alignment" or self.method == "exponentially_decaying_horizon" or self.method == "uniform" else self.buds_phi_alignments[:, :self.start_P], indices=initial_buds_indices)

            if self.fast_sampling:
                max_terms = np.minimum(T + self.P-1, self.l)
                w = np.zeros(shape=(max_terms,)) # enough space for all buds expanded
                w[:self.start_P] = self.buds_pms[:self.start_P]
                self.sample_tree = SumHeap(w)

            non_leaf_buds = np.count_nonzero(self.buds_pms[:self.start_P])

            number_of_buds = self.start_P

            if self.print_last_tree:
                print("We currently have ", non_leaf_buds, " non leaf buds and ", number_of_buds, " number of buds")

            while(number_of_buds < T):
                if self.print_last_tree:
                    print("\n start sampling")
                # Sample from multinomial of buds_pms
                num_samples = min(non_leaf_buds, self.P)

                assert num_samples != 0, "T > l not allowed"

                if self.fast_sampling:
                    buds_samples = self.fast_sampling_without_replacement(num_samples=num_samples)
                else:
                    buds_samples = np.random.choice(number_of_buds, size = num_samples, p=self.buds_pms[:number_of_buds]/np.sum(self.buds_pms[:number_of_buds]), replace=False)

                if self.print_last_tree:
                    if self.method in ["FAVOR+", "FAVOR+ReLU", "positive_alignment", "RandomFourierFeatures_kernel"]:
                        print("Phi Alignments: ", self.buds_phi_alignments[:, :number_of_buds])
                    print("Alignments: ", self.buds_alignments[:, :number_of_buds])
                    print("We have pms: ", self.buds_pms[:number_of_buds]/np.sum(self.buds_pms[:number_of_buds]), "and we have sampled ", buds_samples)
                
                old_indices = self.buds_indices[buds_samples] # Tensor of old indices
                if self.print_last_tree:
                    print("that correspond to indices ", old_indices)
                left_indices = 2 * old_indices + 1 # Tensor of new left indices
                right_indices = 2 * old_indices + 2 # Tensor of new right indices
                
                # apply mask to prevent nodes without children becoming buds
                sample_mask = self.num_sub_tree_leaves[0, right_indices] > 0 # If True, bud sample has two children
                num_nodes_with_two_children = np.count_nonzero(sample_mask)

                if num_nodes_with_two_children != num_samples: # there exist nodes with only one child
                    if self.print_last_tree:
                        print("We have sampled a node with only a left child")
                    # replaces parent with left only child
                    self.buds_indices[buds_samples[~sample_mask]] = left_indices[~sample_mask]

                    # reset probability mass that was set to zero when sampling (without replacement)
                    if self.fast_sampling:
                        for i in buds_samples[~sample_mask]:
                            self.sample_tree.update(i, self.buds_pms[i])

                    left_indices = left_indices[sample_mask]
                    right_indices = right_indices[sample_mask]
                    buds_samples = buds_samples[sample_mask]

                    num_samples = num_nodes_with_two_children


                # only execute if there are still children nodes to compute
                if len(left_indices) > 0:
                    if self.print_last_tree:
                        print("We are expanding ", old_indices, " into ", left_indices, " and ", right_indices)
                    selected_keys = self.K[:, right_indices, :]
                    alignments_right = np.sum(q * selected_keys, axis=-1)
                    alignments_left = self.buds_alignments[:, buds_samples] - alignments_right
                    if self.method == "FAVOR+ReLU" or self.method == "FAVOR+" or self.method == "RandomFourierFeatures" or self.method == "positive_alignment":
                        selected_phi_keys = self.phi_K[:, right_indices, :]
                        phi_alignments_right = np.sum(self.phi_q * selected_phi_keys, axis=-1)
                        phi_alignments_left = self.buds_phi_alignments[:, buds_samples] - phi_alignments_right
                    
                    # adjust only for the masked indices
                    pms_left = self.get_pm(alignments=alignments_left if self.method == "alignment"  or self.method == "exponentially_decaying_horizon" or self.method == "uniform" else phi_alignments_left, indices=left_indices)
                    pms_right = self.get_pm(alignments=alignments_right if self.method == "alignment" or self.method == "exponentially_decaying_horizon" or self.method == "uniform" else phi_alignments_right, indices=right_indices)

                    # replace parent nodes by left child nodes
                    self.buds_indices[buds_samples] = left_indices
                    self.buds_alignments[:, buds_samples] = alignments_left
                    self.buds_pms[buds_samples] = pms_left
                    if self.method == "FAVOR+ReLU" or self.method == "FAVOR+" or self.method == "RandomFourierFeatures" or self.method == "positive_alignment":
                        self.buds_phi_alignments[:, buds_samples] = phi_alignments_left

                    # replace probability mass
                    if self.fast_sampling:
                        for i in buds_samples:
                            self.sample_tree.update(i, self.buds_pms[i])

                    # add right child nodes to buds array
                    self.buds_indices[number_of_buds: number_of_buds + num_samples] = right_indices
                    self.buds_alignments[:, number_of_buds: number_of_buds + num_samples] = alignments_right
                    self.buds_pms[number_of_buds: number_of_buds + num_samples] = pms_right
                    if self.method == "FAVOR+ReLU" or self.method == "FAVOR+" or self.method == "RandomFourierFeatures" or self.method == "positive_alignment":
                        self.buds_phi_alignments[:, number_of_buds: number_of_buds + num_samples] = phi_alignments_right

                    # add new probability mass
                    if self.fast_sampling:
                        for i in range(num_samples):
                            self.sample_tree.update(number_of_buds + i, pms_right[i])

                # update number of buds, and non-leaf buds by removing zero prob nodes (leaves)
                number_of_buds += num_samples
                non_leaf_buds += np.count_nonzero(pms_left) + np.count_nonzero(pms_right) - num_samples

                if self.print_last_tree:
                    print("new buds_indices: ", self.buds_indices[:number_of_buds])

            # take the geometric mean of the values if corresponding indices are not leaf nodes
            #buds_softmax = torch.softmax(self.buds_alignments[:number_of_buds] * self.S / self.num_sub_tree_leaves[self.buds_indices[:number_of_buds]], dim=0).unsqueeze(0)
            buds_exp = np.exp(self.buds_alignments[:, :number_of_buds] * (self.S / self.num_sub_tree_leaves[:, self.buds_indices[:number_of_buds]]))
            buds_softmax = buds_exp / np.sum(buds_exp, axis=-1, keepdims=True)

            selected_values = self.V[:, self.buds_indices[:number_of_buds], :]
            Pred = np.sum(np.expand_dims(buds_softmax, axis=2) * selected_values, axis=1, keepdims=False) # (H, D)

        # update the storage tensors accordingly
        self.post_processing(self.l, k, v)

        self.l += 1

        return torch.Tensor(Pred)

    def fast_sampling_without_replacement(self, num_samples:int):
        buds_samples = np.zeros(shape=(num_samples,), dtype=int)
        for i in range(num_samples):
            buds_samples[i] = self.sample_tree.sample()
            self.sample_tree.update(buds_samples[i], 0) # remove probability mass
        return buds_samples


    def get_pm(self,
        alignments: np.array,
        indices: np.array,
    ):  
        """Calculate Probability Masses (PMs) for alignments based on specified method.

        Args:
            alignments (np.array): Array containing alignments.
            indices (np.array): Array of indices

        Raises:
            Exception: If specified method is not implemented.

        Returns:
            np.array: Array of Probability Masses (PMs) calculated according to specified method 
        """        
        
        num_sub_tree_leaves = self.num_sub_tree_leaves[:, indices] # (1, #indices)

        # + epsilon for stability to not produce zero probabilities for non-leaf nodes.
        if self.method == "alignment" or self.method == "positive_alignment":
            pms = np.sum(np.exp(alignments * (self.S / num_sub_tree_leaves)), axis=0) + self.epsilon # sum probabilities over heads, geometric mean on alignments of subtree leaves
        elif self.method == "FAVOR+ReLU" or self.method == "FAVOR+":
            pms = np.sum(alignments / num_sub_tree_leaves, axis=0) + self.epsilon # sum probabilities over heads, arithmetic mean on alignments of subtree leaves
        elif self.method == "RandomFourierFeatures":
            pms = np.sum(ReLU(alignments) / num_sub_tree_leaves, axis=0) + self.epsilon # sum probabilities over heads, arithmetic mean on alignments of subtree leaves, ReLU for stability (trigo can produce negative results)
        elif self.method == "exponentially_decaying_horizon":
            rightermost_distance_to_l = self.l - self.rightest_leaf_positions[indices] - 1
            leftermost_distance_to_l = self.l - self.leftest_leaf_positions[indices] - 1
            pms = (self.base**rightermost_distance_to_l - self.base**(leftermost_distance_to_l + 1)) + self.epsilon#/(1-self.base) unnecessary to normalize
        elif self.method == "uniform":
            pms = np.ones(len(indices))
        else:
            raise Exception("method not implemented")
        
        # leaf nodes should get pms 0
        pms[num_sub_tree_leaves[0,:] <= 1] = 0

        return pms
        
    def post_processing(self,
        l: int,
        k: np.array,
        v: np.array
    ):
        """
        After T loops of our algorithm, updates the storage tensors K, V, M by propagating
        the current key, query, value up the respective "tree"

        Args:
            l (int): Current 0-indexed token position
            k (torch.Tensor): Current query (at position l)
            v (torch.Tensor): Current value (at position l)
        """
        k = np.squeeze(k, axis=1) # (H, 1, D) -> (H, D)
        v = np.squeeze(v, axis=1) # (H, 1, D) -> (H, D)

        if self.method == "FAVOR+ReLU" or self.method == "FAVOR+" or self.method == "RandomFourierFeatures" or self.method == "positive_alignment":
            self.phi_k = np.squeeze(self.phi_k, axis=1)

        i = self.L-1+l # leaf node index
        for _ in range(0, int(math.log2(self.L)+1)):
            self.K[:, i] += k
            self.V[:, i] += v
            self.num_sub_tree_leaves[0, i] += 1

            if self.method == "FAVOR+ReLU" or self.method == "FAVOR+" or self.method == "RandomFourierFeatures" or self.method == "positive_alignment":
                self.phi_K[:, i, :] += self.phi_k
            if self.method == "exponentially_decaying_horizon":
                if self.leftest_leaf_positions[i] == -1:
                    self.leftest_leaf_positions[i] = l
                self.rightest_leaf_positions[i] = l

            i = (i-1)//2

    def kernel_setup(self, 
        k: np.array, 
        q: np.array, 
    ):
        """Set up kernels for alignment methods.

        Args:
            k (np.array): Array for 'k'
            q (np.array): Array for 'q'
        """        
        H, _, D = q.shape
        if self.method == "FAVOR+":
            self.W = get_proj(R = self.R, D = self.D, orthogonal=True, hyperbolic = True if self.method != "RandomFourierFeatures" else False)
            self.phi_k = kernel(x=k, projection=self.W, method=self.method, exponent=1, scaling=1)
            self.phi_q = kernel(x=q, projection=self.W, method=self.method, exponent=1, scaling=1)
        elif self.method == "RandomFourierFeatures": # scale keys and queries to unit norm to drastically reduce variance
            self.W = get_proj(R = self.R, D = self.D, orthogonal=True, hyperbolic = True if self.method != "RandomFourierFeatures" else False)
            self.phi_k = kernel(x=k, projection=self.W, method=self.method, exponent=1, scaling=1)
            self.phi_q = kernel(x=q, projection=self.W, method=self.method, exponent=1, scaling=1)
        elif self.method == "FAVOR+ReLU":
            self.W = get_proj(R = self.R, D = self.D, orthogonal=True, hyperbolic = True, constant_norm=True)
            self.phi_k = kernel(x=k, projection=self.W, method=self.method, exponent=1, scaling=1)
            self.phi_q = kernel(x=q, projection=self.W, method=self.method, exponent=1, scaling=1)
        elif self.method == "positive_alignment":
            self.phi_k = kernel(x=k, projection=None, method=self.method, exponent=1, scaling=1)
            self.phi_q = kernel(x=q, projection=None, method=self.method, exponent=1, scaling=1)


def ReLU(x):
    return x * (x > 0)

def kernel(x, projection, method, exponent=1, scaling=1):
    """Calculate kernels according to set method

    Args:
        x (np.array): Input array
        projection (np.array or None): Projection array if applicable to the method, else None.
        method (str): Specifies method for kernel calculation
        exponent (int, optional): Exponent value for kernel computation, defaults to 1
        scaling (int, optional): Scaling factor for inner products, defaults to 1 

    Raises:
        AssertionError: If dimensions between 'x' and 'projection' do not match.
        Exception: If the specified kernel method does not exist.

    Returns:
        np.array: Array containing calculated kernels 
    """    
    H, _, D = x.shape
    if method != "positive_alignment":
        D2, R = projection.shape
        assert D == D2, "x and projection should have matching dimensions"
        Wx = np.matmul(np.reshape(x, newshape=(H, D)) * scaling**.5 / D**.25, projection) # also applies scaling for inner products
        assert Wx.shape == (H, R)
        Wx = np.expand_dims(Wx, axis=1) # (H, 1, R)
    if method == "FAVOR+ReLU":
        return ReLU(Wx)**exponent / R**.5
    #elif method == "GeLU_kernel":
    #    return (torch.nn.GELU()(Wx)**exponent) / R**.5
    elif method == "FAVOR+":
        assert exponent==1, "exponent does not affect FAVOR+"
        return np.exp(Wx - np.linalg.norm(x * scaling**.5 / D**.25, ord=2, axis=-1, keepdims=True)**2 / 2) / R**.5
    elif method == "RandomFourierFeatures":
        assert exponent==1, "exponent does not affect RandomFourierFeatures"
        return np.concatenate((np.sin(Wx), np.cos(Wx)), axis=2) * np.exp(np.minimum(np.linalg.norm(x * scaling**.5 / D**.25, axis=2, keepdims=True), 10)**2 / 2) / R**.5 # clamp for stability, e^{10^2/2} = 5E21 is huge anyways
    elif method == "positive_alignment":
        assert exponent==1, "exponent does not affect positive_alignments_kernel"
        return np.concatenate((ReLU(x), ReLU(-x)), axis=2)
    else:
        raise Exception("specified kernel does not exist")
    
def get_proj(R, D, orthogonal=True, hyperbolic=True, constant_norm = False):
    """Generate a projection matrix for specified dimensions and properties.

    Args:
        R (int): Number of output dimensions.
        D (int): Number of input dimensions.
        orthogonal (bool, optional): Determines if the projection matrix should be orthogonal, defaults to True.
        hyperbolic (bool, optional): Determines if the projection is hyperbolic, defaults to True.
        constant_norm (bool, optional): Determines if the projection uses a constant norm instead of chi distribution
    
    Raises:
        AssertionError: If conditions for hyperbolic or orthogonal projections are not met.
    
    Returns:
        np.array: Projection matrix with specified properties based on the given dimensions and options.

    """
    if hyperbolic:
        assert R%2 == 0, "R must be a multiple of two for hyperbolic projection"
        R //= 2
    W = np.random.randn(D, R)

    if orthogonal:
        assert R <= D, "orthogonal projection only works for R <= D"
        if not constant_norm:
            chis = np.linalg.norm(W, ord=2, axis=0,keepdims=True)
        W, _ = np.linalg.qr(W, mode = 'reduced')
        if constant_norm:
            W *= D**.5
        else:
            W = W * chis

    if hyperbolic: 
        W = np.concatenate((W, -W), axis=1)
        R *= 2

    assert W.shape == (D, R), W.shape
    return W


# run to test if gives correct result
if __name__ == "__main__":
    import cProfile, pstats
    import time

    # extra parameters to set
    verbose = True
    line_profiler = False
    test_example = True

    # hyperparameters for algorithm
    D = 64
    L = 1024
    R = 64
    S = 1/math.sqrt(D)
    H = 12
    P = 32
    E = 0.5
    base = 0.99

    if test_example:
        L = 8
        H = 1
        P = 1
        E = 1.0

    a = attention_kernel(
        D=D,
        S=S,
        P=P,
        L=L,
        H=H,
        method="RandomFourierFeatures",
        R=R,
        base=base,
        fast_sampling=True,
        verbose=verbose,
    )
    
    tensor = torch.randn(H, L, 3, D)

    t_max = L-1

    Q = tensor[:, :, 0, :]
    K = tensor[:, :, 1, :]
    V = tensor[:, :, 2, :]

    if test_example:
        # set the query equal to the third key
        Q[:, L-1, :] = K[:, 3, :]

    start1 = time.perf_counter()

    if line_profiler:
        profiler = cProfile.Profile()
        profiler.enable()  
    
    for t in range(t_max+1):
        y_our = a.forward(
            T=math.ceil(t**E),
            k=K[:, t, :],
            q=Q[:, t, :],
            v=V[:, t, :],
        )
    #print(y_our)

    elapsed1 = time.perf_counter()

    if line_profiler:
        profiler.disable()

    start2 = time.perf_counter()
    for t in range(t_max+1):
            A = torch.sum(Q[:, t, :][:, np.newaxis, :] * K[:, :t, :], dim=-1) * S
            a = torch.exp(A)
            softmax = a / torch.sum(a, dim=-1, keepdim=True)
            y = torch.sum(softmax[:, :, np.newaxis] * V[:, :t, :], dim=1)
    elapsed2 = time.perf_counter()
            
    #print("time for scaled dot product attention: ", elapsed2-start2)
    #print("time for casinoatens fast sampling: ", elapsed1-start1)
    #print("time for casinoatens slow sampling: ", elapsed3-start3)

    #print("difference in norm between casinoatens (ceil(t_max^E) terms) and scaled dot product attention (t_max terms): \n", torch.norm(y-y_our))
    #print("difference in norm between fast (ceil(t_max^E) terms) and slow sampling (ceil(t_max^E)) terms): \n", torch.norm(y_our_slow-y_our))

    if line_profiler:
        stats = pstats.Stats(profiler).sort_stats('cumtime')
        stats.print_stats()

