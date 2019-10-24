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
element = MixedElement([Vel, P])
M_ns = FunctionSpace(mesh, element)

M_leray = FunctionSpace(mesh,element)

m_ns = Function(M_ns)
m_l  = Function(M_leray)


dm_ns = TrialFunction(M_ns)
ddm_ns= TestFunction(M_ns)
# Define the Zero initial condition
m0_ns = Function(M_ns)
# for second order time discretization
m0t_ns =Function (M_ns)


# Strain Rate
D = lambda v : (grad(v).T + grad(v)) / 2
# Spin tensor
Spin = lambda v: (grad(v) - grad(v).T)/2

dm_l = TrialFunction(M_leray)
ddm_l= TestFunction(M_leray)


# tuple Trial Function of velocity and pressure = split(m)
(du,dp) = split(dm_ns)
# tuple Test Function of velocity and pressure =split(dm)
(ddv,ddp) = split(ddm_ns)

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

# Inflow Condition for NS
#inflow_profile = Expression(('4.0*1.5*x[1]*(ymax - x[1]) / pow(ymax, 2)', '0'),ymax=1, degree=2)
inflow_profile = Expression(('1.0/(h*h)*(x[1]-h)*(x[1]+h)*(-1)','0.0'), h=1., degree=2)

bcs = [DirichletBC(M_ns.sub(0), inflow_profile  , boundaries, 4 )]
bcs.append(DirichletBC(M_ns.sub(0),Constant((0.,0.)),boundaries, 1 )) 
bcs.append(DirichletBC(M_ns.sub(0),Constant((0.,0.)),boundaries, 2 )) 

bcs_l = [DirichletBC(M_leray.sub(0), inflow_profile  , boundaries, 4 )]
bcs_l.append(DirichletBC(M_leray.sub(0),Constant((0.,0.)),boundaries, 1 )) 
bcs_l.append(DirichletBC(M_leray.sub(0),Constant((0.,0.)),boundaries, 2 )) 



def f(m_ns, time_instant):
    v, p = split(m_ns)
    v_bar,_ = split(m_l) 
    f = + inner( ddp         , div(v)              ) * dx \
        + inner( ddv         , rho * grad(v) * v_bar ) * dx \
        - inner( div(ddv) , p                      ) * dx \
        + inner( D(ddv)      , 2*mu * D(v)         ) * dx \
        - inner( ddv         , b                      ) * dx \
        - inner( ddv         , t                      ) * ds(1)
    return f
# G(dot_m)
def g(dmdt, time_instant):
    dvdt, dpdt  = split(dmdt)
    g =  inner( ddv         , rho * dvdt ) * dx
    return g

# overall form
F = odeint.g_(g, m_ns, m0_ns, m0t_ns) + odeint.f_(f, m_ns, m0_ns)
# linearisation
J = derivative(F, m_ns, dm_ns)
# Boundary conditions / Dirichlet
# TODO : add Stabilization code SUPG, PSPG and SD 
# Define Leray Problem now for filtering 

delta = 0.1
N = 1

# tuple Trial Function of velocity and pressure = split(m)
(dv_bar,dp_bar) = split(dm_l)
# tuple Test Function of velocity and pressure =split(dm)
(ddv_bar,ddp_bar) = split(ddm_l)

#v_bar.interpolate(Expression(('sin(x[0])','0.0'),degree=2))    
#v_bar = interpolate(Expression(('0.','0.0'),degree=2), M_leray.sub(0).collapse())

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


(v, p) = split(m_ns)
(v_bar, p_bar) = split(m_l)

F_leray =inner(ddp_bar, div(dv_bar))*dx \
    -inner(div(ddv_bar),dp_bar)*dx \
    +inner(D(ddv_bar), 2*delta**2*indicator_Q_criteria(v,delta)*D(dv_bar))*dx \
    +inner(ddv_bar,v)*dx  \
    -inner(ddv_bar,dv_bar)*dx 

        

file_handler_press = XDMFFile('output/press.xdmf')
file_handler_vel = XDMFFile('output/vel.xdmf')
file_handler_vel_bar = XDMFFile('output/vel_bar.xdmf')
file_handler_press_bar = XDMFFile('output/press_bar.xdmf')
file_handler_indicator = XDMFFile('output/indicator.xdmf')

# process time steps
for i in range(20):

    # set current time
    t = i * odeint.dt
    #if i%10:
    print("Processing time instant = %4.3f in step %d " % (t,i), end='\n')
    #print("Processing time instant = {} in step {} ").format(t,i)
    # solve nonlinear problem
    solve(F == 0, m_ns, bcs, J=J) # PETScSNES, set option.. 
    

    # extract solution
    v, p = m_ns.split(True) # deepcopy.
    
    indicator = project(indicator_Q_criteria(v,delta))
    indicator.rename('a_Q', 'a_Q')
    # TODO: put parameter for do not save mesh at each time step.
    file_handler_indicator.write(indicator,float(t))
    
    # here we need to check indicator what happen.
    solve(lhs(F_leray) == rhs(F_leray), m_l, bcs_l ) 

    
    v_bar, p_bar = m_l.split(True) # deepcopy.

    # write output
    v.rename('vel', 'vel')
    p.rename('press', 'press')
    # TODO: put parameter for do not save mesh at each time step.
    file_handler_press.write(p,float(t))
    file_handler_vel.write(v,  float(t))
    
    v_bar.rename('vel_bar', 'vel_bar')
    p_bar.rename('press_bar', 'press_bar')
    # TODO: put parameter for do not save mesh at each time step.
    file_handler_press_bar.write(p_bar,float(t))
    file_handler_vel_bar.write(v_bar,  float(t))
 
    
    # update solution states for time integration
    m0_ns, m0t_ns = odeint.update( m_ns, m0_ns, m0t_ns )

print('\nDone.')

















