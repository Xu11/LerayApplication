from dolfin import *

from odeint import * 
# Mesh
nx=50
ny=10

#parameters["mesh_partitioner"] = "parmetis"

# structured Mesh.
mesh = RectangleMesh(Point(0.0, 0.0), Point(10., 1.0), nx, ny,"right")
#
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
boundaries.set_all(0)
Top = CompiledSubDomain("x[1]==side && on_boundary",side=1.) 
Top.mark(boundaries,1)
Bottom = CompiledSubDomain("x[1]==side && on_boundary",side=0.) 
Bottom.mark(boundaries,2)
Left = CompiledSubDomain("x[0]==side && on_boundary",side=0.) 
Left.mark(boundaries,3)
Right = CompiledSubDomain("x[0]==side && on_boundary",side=10.) 
Right.mark(boundaries,4)

dx = Measure('dx',domain=mesh)
ds = Measure('dx',domain=mesh, subdomain_data=boundaries)
# Define the function space
# Function space
Vel = VectorElement("Lagrange", mesh.ufl_cell(), 2) #P2 
P = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #P1
element = MixedElement([Vel, Vel,Vel, P, P])
M = FunctionSpace(mesh, element)

m = Function(M)

dm = TrialFunction(M)
ddm= TestFunction(M)
# Define the Zero initial condition
m0 = Function(M)
# for second order time discretization
m0t =Function (M)
# Strain Rate
D = lambda v : (grad(v).T + grad(v)) / 2
# Spin tensor
Spin = lambda v: (grad(v) - grad(v).T)/2
# tuple Trial Function of velocity and pressure = split(m)
(du,du_bar,du_conv,dp,dlmbda) = split(dm)
# tuple Test Function of velocity and pressure =split(dm)
(ddv,ddv_bar,ddv_conv,ddp,ddlmbda) = split(ddm)

# Constant Value
# Source Term
b = Constant((0.,0.))
# Neumann traction
t = Constant((0.,0.))
# Time integration setup
odeint = odeint(dt = 0.01, rho=0.8) # make it a bit dissipative
# if i  need to adapt one just do this 
odeint.dt = 0.1
# material parameter
rho = Constant(1.)
mu  = Constant(0.001)


inflow_profile = Expression(('4.0*1.5*x[1]*(ymax - x[1]) / pow(ymax, 2)', '0'),ymax=1, degree=2)
bcs = [DirichletBC(M.sub(0), inflow_profile  , boundaries, 4 )]
bcs.append(DirichletBC(M.sub(1),Constant((0.,0.)),boundaries, 0 )) 
bcs.append(DirichletBC(M.sub(1),Constant((0.,0.)),boundaries, 1 )) 
bcs.append(DirichletBC(M.sub(1),Constant((0.,0.)),boundaries, 2 )) 
bcs.append(DirichletBC(M.sub(1),Constant((0.,0.)),boundaries, 3 )) 
bcs.append(DirichletBC(M.sub(0),Constant((0.,0.)),boundaries, 1 )) 
bcs.append(DirichletBC(M.sub(0),Constant((0.,0.)),boundaries, 2 )) 



def indicator_Q_criteria(u,delta):
    # comute Q(u,u)
    Q = 0.5*inner(Spin(u),Spin(u)) \
        - inner(grad(u), grad(u))
    a_Q = 0.5 - 1/pi*atan(1/delta*Q/(abs(Q)+delta*delta))

    return a_Q

def indicator_Vreman(u):
    # Note: grad(v)[i,j] = v[i].dx(j)
    if u.geometric_dimension() ==2 :
        B =grad(v)[0,0]*grad[1,1]- grad(v)[0,1]**2
    else:
        B = grad(v)[0,0]*grad[1,1] \
            +grad(v)[0,0]*grad[2,2] \
            +grad(v)[1,1]*grad[2,2] \
            -grad(v)[0,1]**2 \
            -grad(v)[1,2]**2 \
            -grad(v)[0,2]**2 
    # Frob sum_i sum_j grad(u)_i,j
    frobNorm = trace(inner(grad(u),grad(u)))
    a_v = sqrt(B/frobNorm**2)
    return a_v

def indicator_H(u,delta):
    w = curl(u)
    a_H = (1- abs(inner(u,w)/ (inner(abs(u),abs(w))+delta**2)))

    return a_H

#def indicator_D(u_conv,delta,N):
#   a_D = (-1)**(N-1)* delta**(2*N+2)

def f(m, time_instant):
    delta = 0.0001
    N = 1
    v,v_bar,v_conv, p,lmbda = split(m)
    
    indicator= (-1)**(N-1)* delta**(2*N+2)*v_conv

    f = + inner( ddp         , div(v)              ) * dx \
        + inner( ddv         , rho * grad(v) * v_bar ) * dx \
        - inner( div(ddv) , p                      ) * dx \
        + inner( D(ddv)      , 2*mu * D(v)         ) * dx \
        - inner( ddv         , b                      ) * dx \
        - inner( ddv         , t                      ) * ds(1)
    # here i add Leray Filter
    f+= inner(ddlmbda, div(v_bar))*dx \
        -inner(div(ddv_bar),lmbda)*dx \
        +inner(D(ddv_bar),inner(indicator,indicator)*D(v_bar))*dx \
        +inner(ddv,v-v_bar)*dx
    
    f+= inner(v_conv,ddv_conv)*dx+\
        delta**2*inner(grad(v_conv),grad(ddv_conv))*dx

    return f
# G(dot_m)
def g(dmdt, time_instant):
    dvdt, dvbardt, dvconvdt, dpdt, dlmbdadt  = split(dmdt)
    g =  inner( ddv         , rho * dvdt ) * dx
    return g

# overall form
F = odeint.g_(g, m, m0, m0t) + odeint.f_(f, m, m0)
# linearisation
J = derivative(F, m, dm)
# Boundary conditions / Dirichlet

file_handler_press = XDMFFile('output/press.xdmf')
file_handler_vel = XDMFFile('output/vel.xdmf')

# process time steps
for i in range(20):

    # set current time
    t = i * odeint.dt
    #if i%10:
    print("Processing time instant = %4.3f in step %d " % (t,i), end='\n')
    #print("Processing time instant = {} in step {} ").format(t,i)
    # solve nonlinear problem
    solve(F == 0, m, bcs, J=J) # PETScSNES, set option.. 
    # extract solution
    v_, p_ = m.split(True) # deepcopy.
    # write output
    v_.rename('vel', 'vel')
    p_.rename('press', 'press')
    # TODO: put parameter for do not save mesh at each time step.
    file_handler_press.write_checkpoint(p_,"press", float(t))
    file_handler_vel.write_checkpoint(v_, "vel", float(t))
    # update solution states for time integration
    m0, m0t = odeint.update( m, m0, m0t )

print('\nDone.')






