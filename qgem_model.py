import numpy as np
from numpy import linalg
import random

def Ejk(j,k, dimension):
    matrix = np.zeros((dimension, dimension))
    matrix[j][k] = 1
    return matrix


def generalised_paulis_list(dimension):
    # source: http://mathworld.wolfram.com/GeneralizedGell-MannMatrix.html
    gen_paulis_list = []
    gen_paulis_list.append(np.identity(dimension))
    j = 0
    k = 0
    factor = np.sqrt(dimension/2.) #adapted from PhysRevA.83.032318
    for k in range(dimension):
        for j in range(dimension):
             if k<j:
                 gen_paulis_list.append(factor*(Ejk(j,k, dimension) + Ejk(k,j, dimension)))
             if k>j:
                 gen_paulis_list.append(-1.j*factor*(Ejk(j,k, dimension) - Ejk(k,j, dimension)))  
    for l in range(1, dimension):
        blank = np.zeros((dimension, dimension))
        for j in range(1, l + 1):
            blank = blank + Ejk(j-1,j-1, dimension) 
        blank = blank - (l)*Ejk(l,l, dimension)
        gen_paulis_list.append(factor*np.sqrt((2/(l*(l+1))))*blank)
    return gen_paulis_list

def gen_paulis_tensors(dimension, num_qudits):
    paulis_d = generalised_paulis_list(dimension)
    if num_qudits == 1:
        return paulis_d
    else:
        new_level = []
        next_level = gen_paulis_tensors(dimension, num_qudits - 1)
        for i in range(len(paulis_d)):
            for j in range((dimension**2)**(num_qudits - 1)):  
                new_level.append(np.kron(paulis_d[i], next_level[j]))
        return new_level

def gen_paulis_tensors_diag(dimension, num_qudits=2):
    paulis_d = generalised_paulis_list(dimension)
    diag_list = []
    for j in range(len(paulis_d)):  
        diag_list.append(np.kron(paulis_d[j], paulis_d[j]))
    return diag_list

def create_EW_list(dimension):
    paulis = gen_paulis_tensors_diag(2)
    list_EW =[]
    for j in range (len(paulis)):
        list_EW.append(EW(paulis[j]))
    return list_EW


def apply_operators(dimension, q_state, physics_values, operator_list):
    expect_values = [] #WHY? also I gt negative values then so what is happening?
    for operator in operator_list:
        witness = EW(operator)
        expect_values.append(witness.apply_to_state(q_state, physics_values))
    return expect_values

def create_EW_from_letters(paulis_list_letters):
    paulis_list = []
    for term in paulis_list_letters:
        paulis_list.append(Q_state.make_EW_letters(2, [term], weights=[1]))
    return paulis_list

class Physics:
    def __init__(self, gravity, h_bar):
        self.gravity = gravity
        self.h_bar = h_bar

class Q_state:
    """ Defines the quantum state for the experiement, model values must be specified at
        initialisation - use pre-print as reference; decoherence rate and time must be specified 
        after initialisation"""
    def __init__(self, dimension, distance, delta_x, m1, m2, theta1=0., theta2=0.):
        self.dimension = dimension
        self.distance = distance
        self.delta_x = delta_x
        self.m1 = m1
        self.m2 = m2
        self.gamma = 0.
        self.time = 0.
        self.theta1 = theta1
        self.theta2 = theta2

    def vectorise(self, physics_values):
        """ Return a normalised vector representing the quantum state """
        state_vector = np.zeros((self.dimension**2, 1), dtype=np.complex_)
        for i in range(self.dimension):
            for j in range(self.dimension):
                length_1 = (self.dimension -1 -i)*self.delta_x/(self.dimension - 1)
                length_2 = np.sqrt(self.distance**2 + ((j*self.delta_x)/(self.dimension - 1.))**2 - 2*self.distance*((j*self.delta_x)/(self.dimension - 1.))*np.cos(np.pi - self.theta2))
                separation = np.sqrt(length_1**2 + length_2**2 - 2*length_1*length_2*np.cos(np.pi - self.theta1 + np.arcsin((j*(self.delta_x/(self.dimension -1))*np.sin(self.theta2))/length_2)))
                phase = physics_values.gravity*self.m1*self.m2*self.time/(physics_values.h_bar*separation)
                state_vector[i*self.dimension + j] = np.exp(1.0j*phase)
        return (1/self.dimension)*state_vector      

    def density_matrix_decohere(self, physics_values):
        """ Return the density matrix of the state, taking into account decoherence"""
        state_vector = self.vectorise(physics_values)
        decoherence_mask = np.identity(self.dimension)
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    decoherence_mask[i][j] = np.exp(-1.*self.gamma*self.time)
        decoherence_mask = np.kron(decoherence_mask, decoherence_mask)
        density_matrix = np.kron(state_vector, state_vector.conj().T) * decoherence_mask
        return density_matrix

    def density_matrix(self, physics_values):
        """ Return the density matrix of the state, without taking into account decoherence"""
        state_vector = self.vectorise(physics_values)
        density_matrix = np.kron(state_vector, state_vector.conj().T)
        return density_matrix

    def von_neuman_entropy(self, physics_values):
        """ Compute bipartite VNE for the state speficied """
        density_mat = self.density_matrix_decohere(physics_values)
        partial_trace = np.ndarray(shape=(self.dimension, self.dimension), dtype=np.complex_)
        for row_block in range(0, self.dimension):
            for column_block in range(0, self.dimension):
                mat_element = 0
                for i in range(row_block*self.dimension, (row_block + 1)*self.dimension):
                    for j in range(column_block*self.dimension, (column_block+1)*self.dimension):
                        if i%self.dimension == j%self.dimension:
                            mat_element = mat_element + density_mat[i][j] 
                partial_trace[row_block][column_block] = mat_element   
        evalsh, evectsh = np.linalg.eigh(partial_trace)
        vn_entropy = 0
        for value in evalsh:
            if value > 0: 
                vn_entropy = vn_entropy - value*np.log2(value)
        return vn_entropy
    
    @staticmethod
    def partial_transpose(dimension, density_mat):
        """ returns the partial transpose of a density matrix - note this is a static method """
        pt_mat = np.zeros((dimension**2, dimension**2), dtype=np.complex_)
        for row_block in range(dimension):
            for column_block in range(dimension):
                for i in range(dimension):
                    for j in range(dimension):
                        pt_mat[i + row_block*dimension][j + column_block*dimension] = density_mat[j + row_block*dimension][i + column_block*dimension]
        return pt_mat

    def is_entangled(self, physics_values):
        """ Not very reliable function to quickly test if a state is NPT """
        pt_mat = Q_state.partial_transpose(self.dimension, self.density_matrix(physics_values))
        eigvals = np.linalg.eigvals(pt_mat)
        for value in eigvals:
            if value<-1e-16:
                return "Yes" 
        if self.dimension ==2: 
            return "No"
        else:
            return "Unknown"

    def find_PPT_EW(self, physics_values):
        """ Returns the matrix form of the PPT entanglement witness for a given state """
        pt_mat = Q_state.partial_transpose(self.dimension, self.density_matrix(physics_values))
        evals, evects = np.linalg.eigh(pt_mat)
        min_eval = np.where(evals == np.amin(evals))
        vect = evects[:,min_eval]
        vect = evects[:,min_eval].reshape(self.dimension**2,1)
        if evals[min_eval] <-1e-15:
            eigenstate_density = np.kron(vect, vect.conj().T)
            witness = Q_state.partial_transpose(self.dimension, eigenstate_density)
            return witness  
        else:
            return False

    def find_PPT_EW_deco(self, physics_values):
        """ Returns the matrix form of the PPT entanglement witness for a given state for a given 
        decoherence rate - though note this is for test purposes only """
        pt_mat = Q_state.partial_transpose(self.dimension, self.density_matrix_decohere(physics_values))
        evals, evects = np.linalg.eigh(pt_mat)
        min_eval = np.where(evals == np.amin(evals))
        vect = evects[:,min_eval]
        vect = evects[:,min_eval].reshape(self.dimension**2,1)
        if evals[min_eval] <-1e-15:
            eigenstate_density = np.kron(vect, vect.conj().T)
            witness = Q_state.partial_transpose(self.dimension, eigenstate_density)
            return witness  
        else:
            eigenstate_density = np.kron(vect, vect.conj().T)
            witness = Q_state.partial_transpose(self.dimension, eigenstate_density)
            return witness 

    def find_schmidt_EW(self, physics_values):
        """ Returns the matrix form of the Schmidt entanglement witness for a given state """
        density_mat = self.density_matrix_decohere(physics_values)
        partial_trace_B = np.ndarray(shape=(self.dimension, self.dimension), dtype=np.complex_)
        for row_block in range(0, self.dimension):
            for column_block in range(0, self.dimension):
                mat_element = 0
                for i in range(row_block*self.dimension, (row_block + 1)*self.dimension):
                    for j in range(column_block*self.dimension, (column_block+1)*self.dimension):
                        if i%self.dimension == j%self.dimension:
                            mat_element = mat_element + density_mat[i][j] 
                partial_trace_B[row_block][column_block] = mat_element   
        partial_trace_A = np.ndarray(shape=(self.dimension, self.dimension), dtype=np.complex_)  
        for i in range(0, self.dimension):
            for j in range(0, self.dimension):
                mat_element = 0
                for row_block in range(0, self.dimension):
                    for column_block in range(0, self.dimension):
                        if row_block == column_block:
                            mat_element = mat_element + density_mat[i + self.dimension*row_block][j + self.dimension*column_block] 
                partial_trace_A[i][j] = mat_element 
        evalsh_B, evectsh_B = np.linalg.eigh(partial_trace_B)
        evalsh_A, evectsh_A = np.linalg.eigh(partial_trace_A) 
        witness = np.eye(self.dimension**2, dtype=np.complex_)
        for dim in range(self.dimension):
            mat1 = np.outer(evectsh_B[:,dim].conj().T, evectsh_B[:,dim])
            mat2 = np.outer(evectsh_A[:,dim].conj().T, evectsh_A[:,dim])
            tens = np.kron(mat1, mat2)
            witness = witness - tens
        return witness         

    def find_vicinity_EW(self, physics_values):
        """ Returns the matrix form of the Vicinity entanglement witness for a given state """
        density_mat = self.density_matrix_decohere(physics_values)
        partial_trace_B = np.ndarray(shape=(self.dimension, self.dimension), dtype=np.complex_)
        for row_block in range(0, self.dimension):
            for column_block in range(0, self.dimension):
                mat_element = 0
                for i in range(row_block*self.dimension, (row_block + 1)*self.dimension):
                    for j in range(column_block*self.dimension, (column_block+1)*self.dimension):
                        if i%self.dimension == j%self.dimension:
                            mat_element = mat_element + density_mat[i][j] 
                partial_trace_B[row_block][column_block] = mat_element   
        partial_trace_A = np.ndarray(shape=(self.dimension, self.dimension), dtype=np.complex_)
        evalsh_B, evectsh_B = np.linalg.eigh(partial_trace_B)
        coef = max(evalsh_B)
        id_mat = np.eye(self.dimension**2, dtype=np.complex_)
        witness = coef*id_mat - density_mat
        return witness

    @staticmethod
    def make_EW_operators(dimension, operators, weights):
        """ Returns a matrix form for a given set of matrix operators and weights """
        witness = np.zeros((dimension**2, dimension**2))
        for operator, weight in zip(operators, weights):
            witness = witness + weight*operator
        return witness

    @staticmethod
    def make_EW_letters(dimension, operators, weights):
        """ Returns a matrix form for a given set of Pauli strings and weights """
        paulis = generalised_paulis_list(dimension)
        witness = 0
        for operator, weight in zip(operators, weights):
            temp_witness = 1
            for letter in operator:
                if letter=='I':
                    temp_witness = np.kron(temp_witness, paulis[0])
                if letter=='X':
                    temp_witness = np.kron(temp_witness, paulis[1])
                if letter=='Y':
                    temp_witness = np.kron(temp_witness, paulis[2])
                if letter=='Z':
                    temp_witness = np.kron(temp_witness, paulis[3])
            witness = witness + weight*temp_witness
        print(witness)
        return witness


    def get_measurement_probas_cumul(self, measurement_operator, physics_values):
        probas = []
        e_values, e_vects = np.linalg.eig(measurement_operator)
        vector_state = self.vectorise(physics_values)
        for i in range(self.dimension**2):
            eig_vector = e_vects[:, i]
            probas.append(np.square(np.linalg.norm(np.dot(eig_vector.conj(), vector_state))))
        cumul_probas = np.cumsum(probas)
        return cumul_probas, e_values

    def get_measurement_probas_cumul_dens(self, measurement_operator, physics_values):
        probas = []
        e_values, e_vects = np.linalg.eig(measurement_operator)
        density_mat = self.density_matrix_decohere(physics_values)
        for i in range(self.dimension**2):
            eig_vector = e_vects[:, i]
            eig_density = density_matrix = np.outer(eig_vector, eig_vector.conj().T)
            probas.append(np.trace(np.matmul(density_mat, eig_density)))
        cumul_probas = np.cumsum(probas)
        return cumul_probas, e_values


    def get_measurement_probas(self, measurement_operator, physics_values):
        probas = []
        e_values, e_vects = np.linalg.eig(measurement_operator)
        vector_state = self.vectorise(physics_values)
        for i in range(self.dimension**2):
            eig_vector = e_vects[:, i]
            probas.append(np.linalg.norm(np.square(np.dot(eig_vector, vector_state))))
        return probas, e_values


    def single_measurement(self, cumul_probas, e_values):
        mark = random.random()
        count = 0
        for value in cumul_probas:
            if value>=mark:
                return e_values[count] 
            count = count + 1
        return e_values[-1]
        #model measurment noise

    def get_witness_probas(self, witnesss, basis):
        weights = witness.decomposition(basis)
        operator_list = basis
        for weight, elem in zip(weights, operator_list):
            if weight == 0:
                weights.remove(weight)
                operator_list.remove(elem)
        probas_list = []
        for elem in operator_list: 
            probas_list.append(self.get_measurement_probas(self, measurement_operator, physics_values))
        return probas_list, operator_list

class EW:
    """ Defines an Entanglement Witness, the matrix form of which must be specified at 
        initialisation"""
    def __init__(self, matrix_rep):
        self.matrix_rep = matrix_rep

    def apply_to_state(self, q_state, physics_values):
        """ Returns the expectation value of the witness with respect to a given quantum state"""
        density_mat = q_state.density_matrix_decohere(physics_values)
        return np.real(np.trace(np.matmul(self.matrix_rep, density_mat)))

    def decomposition(self, basis):
        """ Returns the decomposition of the witness in terms of Pauli strings and weights"""
        weights = []
        dimension = float(self.matrix_rep.shape[0])
        for operator in basis:
               weight = np.trace(np.matmul(self.matrix_rep, operator))
               if abs(weight) < 1e-12:
                   weight = 0
               weights.append(weight)
        return (1./dimension)*np.array(weights)

