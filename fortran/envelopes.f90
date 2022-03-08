module gauss
use iso_c_binding
implicit none

real*8, parameter :: sq2pi = sqrt(8.d0 * atan(1.d0))

contains

subroutine gaussian(y, x, s, m, j) bind(C, name="gaussian")

    integer (c_int), bind(c) :: j
    real (c_double), bind(c) :: s, m
    real (c_double), dimension(j), bind(c) :: x
    real (c_double), dimension(j), bind(c), intent(out) :: y
    
    real*8, dimension(j) :: r, r2
    
    r = x - m
    y = exp(-0.5d0 * r * r / (s * s))
    
end

subroutine half_truncated_gaussian(y, x, x0, s, m, j) bind(C, name="half_truncated_gaussian")
    
    integer (c_int), bind(c) :: j
    integer :: i
    real (c_double), bind(c) :: s, m, x0
    real (c_double), dimension(j), bind(c) :: x
    real (c_double), dimension(j), bind(c), intent(out) :: y
    
    real*8 :: r
        
    do i=1,j,1
        if (x(i) .GT. x0) then
            r = x(i) - m
            y(i) = exp(-0.5d0 * r * r / (s * s))
        else
            y(i) = 0.d0
        end if
    end do

end

end module gauss