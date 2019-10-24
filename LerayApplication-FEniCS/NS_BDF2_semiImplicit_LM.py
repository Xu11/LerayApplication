import numpy as np
from dolfin import *
from odeint import * 

# material parameter
rho = Constant(1.)
mu  = Constant(0.001)

alpha_0 =  1.5
alpha_1 = -2.0
alpha_2 =  0.5
dt = 0.0025
comm = MPI.comm_world
file_handler_lambda = XDMFFile('output/lambda_implicit_BDF2.xdmf')
file_handler_press = XDMFFile('output/press_implicit_BDF2.xdmf')
file_handler_vel   = XDMFFile('output/vel_implicit_BDF2.xdmf')
file_handler_lift = XDMFFile('output/lift_implicit_BDF2.xdmf')
file_handler_drag = XDMFFile('output/drag_implicit_BDF2.xdmf')
file_handler_pressureDiff = XDMFFile('output/pressureDiff_implicit_BDF2.xdmf')

#parameters["mesh_partitioner"] = "parmetis"

## Mesh=================================================================
# ## Structured mesh------------------------------------------------------
# nx=100
# ny=40
# mesh = RectangleMesh(Point(0.0, 0.0), Point(10., 1.0), nx, ny,"right")
# with XDMFFile(comm, "output/meshRec.xdmf") as xdmf_handler:
#     xdmf_handler.write(mesh)
# boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
# boundaries.set_all(0)
# Top = CompiledSubDomain("x[1]==side && on_boundary",side=1.) 
# Top.mark(boundaries,1)
# Bottom = CompiledSubDomain("x[1]==side && on_boundary",side=0.) 
# Bottom.mark(boundaries,2)
# Left = CompiledSubDomain("x[0]==side && on_boundary",side=0.) 
# Left.mark(boundaries,3)
# Right = CompiledSubDomain("x[0]==side && on_boundary",side=10.) 
# Right.mark(boundaries,4)
# with XDMFFile(comm, "output/boundaryRec.xdmf") as xdmf_handler:
#     xdmf_handler.write(boundaries)

## Read XDMF mesh-------------------------------------------------------
meshPath="mesh/mesh_fine.xdmf"
boundaryMeshPath="mesh/mesh_fine.xdmf"


mesh = Mesh(comm)
with XDMFFile(comm, meshPath) as xdmf:
     xdmf.read(mesh)

boundaries = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
with XDMFFile(comm, boundaryMeshPath) as xdmf:
     xdmf.read(boundaries)

dx = Measure('dx',domain=mesh)
ds = Measure('ds',domain=mesh, subdomain_data=boundaries)
# dsb =Measure('dx',domain=boundaries)
# Define the function space
# Function space
## Element, FunctionSpace, Trial/test functions======================================
Vel = VectorElement("Lagrange", mesh.ufl_cell(), 2) #P2 
P = FiniteElement("Lagrange", mesh.ufl_cell(), 1) #P1
LM = FiniteElement("R",mesh.ufl_cell(), 0) # real element for Lagrange multiplier (one global dof)
element = MixedElement([Vel, P, LM])

M_ns = FunctionSpace(mesh, element)
dm_ns = TrialFunction(M_ns)
ddm_ns= TestFunction(M_ns)

# tuple Trial Function of velocity and pressure = split(m)
(du,dp,dlambda) = split(dm_ns)
# tuple Test Function of velocity and pressure =split(dm)
(ddv,ddp,ddlambda) = split(ddm_ns)

m_ns = Function(M_ns)
# Define the Zero initial condition
m0_ns = Function(M_ns)
# for second order time discretization
m00_ns =Function (M_ns)

M_leray = FunctionSpace(mesh,element)
dm_l = TrialFunction(M_leray)
ddm_l= TestFunction(M_leray)
m_l  = Function(M_leray)

# Strain Rate
D = lambda v : (grad(v).T + grad(v)) / 2
# Spin tensor
Spin = lambda v: (grad(v) - grad(v).T)/2



# Constant Value
# Source Term
b = Constant((0.,0.))
# Neumann traction
t = Constant((0.,0.))
# Time integration setup
# odeint = odeint(dt = 0.01, rho=0.8) # make it a bit dissipative
# if i  need to adapt one just do this 
# odeint.dt = 0.01

# Inflow Condition for NS
#inflow_profile = Expression(('4.0*1.5*x[1]*(ymax - x[1]) / pow(ymax, 2)', '0'),ymax=1, degree=2)
# inflow_profile = Expression(('1.0/(h*h)*(x[1]-h)*(x[1]+h)*(-1)','0.0'), h=1., degree=2)

# bcs.append(DirichletBC(M_ns.sub(0),inflow_profile,boundaries, 3 )) 
# bcs_l = [DirichletBC(M_leray.sub(0), Constant((0.,0.))  , boundaries, 1 )]
# bcs_l.append(DirichletBC(M_leray.sub(0),Constant((0.,0.)),boundaries, 2 )) 
# bcs_l.append(DirichletBC(M_leray.sub(0),Constant((0.,0.)),boundaries, 3 )) 
# bcs_l.append(DirichletBC(M_leray.sub(0),Constant((0.,0.)),boundaries, 4 )) 

# # Ossen problem 
def f(m_ns, m0_ns, m00_ns, alpha_0, alpha_1, alpha_2, dt):
    # v, p  = split(m_ns)
    v0, _,_ = split(m0_ns) # deepcopy.
    v00,_,_ = split(m00_ns)
    v_star= 2*v0 - v00
    # v_bar,_ = split(m_ns) 
    f = + inner( ddp         , div(du)              ) * dx \
        + inner( ddv         , rho * grad(du) * v_star   ) * dx \
        - inner( div(ddv)    , dp                   ) * dx \
        + inner( D(ddv)      , 2*mu * D(du)         ) * dx \
        + alpha_0/dt * inner( ddv, rho * du ) * dx \
        + alpha_1/dt * inner( ddv, rho * v0) * dx \
        + alpha_2/dt * inner( ddv, rho * v00) * dx \
        + dlambda * ddp * dx\
        + ddlambda * dp * dx
        # - inner( ddv         , b                      ) * dx \
        # - inner( ddv         , t                      ) * ds(1)
    return f
# # G(dot_m)
# def g(dmdt, time_instant):
#     dvdt, dpdt  = split(dmdt)
#     g =  inner( ddv         , rho * dvdt ) * dx
#     return g

# overall form
F = f(m_ns, m0_ns, m00_ns, alpha_0, alpha_1, alpha_2, dt)
   
# linearisation
# J = derivative(F, m_ns, dm_ns)
# Boundary conditions / Dirichlet
# TODO : add Stabilization code SUPG, PSPG and SD 
# Define Leray Problem now for filtering 

# delta = 0.1
# N = 1

# tuple Trial Function of velocity and pressure = split(m)
# (dv_bar,dp_bar) = split(dm_l)
# # tuple Test Function of velocity and pressure =split(dm)
# (ddv_bar,ddp_bar) = split(ddm_l)

#v_bar.interpolate(Expression(('sin(x[0])','0.0'),degree=2))    
#v_bar = interpolate(Expression(('0.','0.0'),degree=2), M_leray.sub(0).collapse())

# def indicator_Q_criteria(u,delta):
#     # comute Q(u,u)
#     Q = 0.5*inner(Spin(u),Spin(u)) \
#         - inner(grad(u), grad(u))
#     a_Q = 0.5 - 1/pi*atan(1/delta*Q/(abs(Q)+delta*delta))

#     return a_Q

# def indicator_Vreman(u):
#     # Note: grad(v)[i,j] = v[i].dx(j)
#     if u.geometric_dimension() ==2 :
#         B =grad(v)[0,0]*grad[1,1]- grad(v)[0,1]**2
#     else:
#         B = grad(v)[0,0]*grad[1,1] \
#             +grad(v)[0,0]*grad[2,2] \
#             +grad(v)[1,1]*grad[2,2] \
#             -grad(v)[0,1]**2 \
#             -grad(v)[1,2]**2 \
#             -grad(v)[0,2]**2 
#     # Frob sum_i sum_j grad(u)_i,j
#     frobNorm = trace(inner(grad(u),grad(u)))
#     a_v = sqrt(B/frobNorm**2)
#     return a_v

# def indicator_H(u,delta):
#     w = curl(u)
#     a_H = (1- abs(inner(u,w)/ (inner(abs(u),abs(w))+delta**2)))

#     return a_H

t = 0.
inflow_profile = Expression(('4*Um*(x[1]*(ymax-x[1]))*sin(pi*t/8.0)/(ymax*ymax)', '0'), ymax=0.41,Um = 1.5, t = t, degree=2)

bcs = [DirichletBC(M_ns.sub(0), inflow_profile  , boundaries, 5 )]
bcs.append(DirichletBC(M_ns.sub(0),Constant((0.,0.)),boundaries, 2 )) 
bcs.append(DirichletBC(M_ns.sub(0),Constant((0.,0.)),boundaries, 4 ))
bcs.append(DirichletBC(M_ns.sub(0),Constant((0.,0.)),boundaries, 6 )) 
bcs.append(DirichletBC(M_ns.sub(0),Constant((0.,0.)),boundaries, 7 )) 
bcs.append(DirichletBC(M_ns.sub(0),Constant((0.,0.)),boundaries, 8 )) 
bcs.append(DirichletBC(M_ns.sub(0),Constant((0.,0.)),boundaries, 9 ))
bcs.append(DirichletBC(M_ns.sub(0), inflow_profile  , boundaries, 3 ))

lift_array  = np.empty((0,2),float)
drag_array  = np.empty((0,2),float)
pDiff_array = np.empty((0,2),float)
# process time steps
for i in range(10):

    # set current time
    t = i*dt
    # t = i * odeint.dt
    
    #if i%10:
    print("Processing time instant = %4.3f in step %d " % (t,i), end='\n')
    #print("Processing time instant = {} in step {} ").format(t,i)
    # solve nonlinear problem
    inflow_profile.t = t
    # solve(F == 0, m_ns, bcs, J=J) # PETScSNES, set option.. 
    solve(lhs(F) == rhs(F), m_ns, bcs) # PETScSNES, set option.. 

    # extract solution
    v, p, lambdaValue = m_ns.split(True) # deepcopy.
    
    # TODO: put parameter for do not save mesh at each time step.
    

    # write output
    v.rename('vel', 'vel')
    p.rename('press', 'press')
    # TODO: put parameter for do not save mesh at each time step.
    file_handler_press.write(p,float(t))
    file_handler_vel.write(v,  float(t))
    file_handler_lambda.write(lambdaValue,  float(t))
    # Compute drag and lift
    # n = FacetNormal(mesh)
    # I = Identity(mesh.geometry().dim())
    # T = -p*I + 2.0*mu*sym(grad(v))
    # force = dot(T, n)
    # # D = (force[0]*20)*ds(6)+(force[0]*20)*ds(7)+(force[0]*20)*ds(8)+(force[0]*20)*ds(9)
    # D = (force[0]*20)*(ds(6)+ds(7)+ds(8)+ds(9))
    # L = -(force[1]*20)*(ds(6)+ds(7)+ds(8)+ds(9))
    # drag = assemble(D)
    # lift = assemble(L)
    # info("drag= %e    lift= %e" % (drag , lift))
    # # lift_array = np.vstack((lift_array,np.array([t,lift])))
    # # drag_array = np.vstack((drag_array,np.array([t,drag])))

    # # Compute pressure difference
    # a_1 = Point(0.15, 0.2)
    # a_2 = Point(0.25, 0.2)
    # p_diff = p(a_1) - p(a_2)
    # info("p_diff = %e" % p_diff)
    # pDiff_array = np.vstack((pDiff_array,np.array([t,p_diff])))

    # update solution states for time integration
    # m0_ns, m0t_ns = odeint.update( m_ns, m0_ns, m0t_ns )
    m00_ns.assign(m0_ns)
    m0_ns.assign(m_ns)

print('\nDone.')
if __name__ == "__main__":
   list_timings(TimingClear.clear, [TimingType.wall])
   # print(lift_array)
   # np.savetxt('lift.txt',lift_array,fmt='%1.3f,%1.6e')
   # np.savetxt('drag.txt',drag_array,fmt='%1.3f,%1.6e')
   # np.savetxt('pressDiff.txt',pDiff_array,fmt='%1.3f,%1.6e')
